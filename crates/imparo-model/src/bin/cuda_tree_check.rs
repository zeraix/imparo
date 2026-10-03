//! Bounded native-tree state and cursor gate; uses the existing target workflow.
use imparo_model::{Model, speculative::{DraftProvider, DraftTree}};

#[derive(Clone, Copy)]
struct Case { parents: &'static [i32], depth: &'static [usize], path: &'static [i32], wrong: usize, block: usize }
const LFM2: Case = Case { parents: &[-1,0,1,2,1,4], depth: &[0,1,2,3,2,3], path: &[0,1,4,5], wrong: 2, block: 4 };
const E4B: Case = Case { parents: &[-1,0,1,0], depth: &[0,1,2,1], path: &[0,3], wrong: 1, block: 3 };
const E4B_DELAYED: Case = Case { parents: &[-1,0,1,1], depth: &[0,1,2,2], path: &[0,1,3], wrong: 2, block: 3 };
const EMIT: usize = 128;

fn tree(case: Case, trail: &[u32], at: usize, vocab: u32) -> DraftTree {
    let tokens = case.depth.iter().enumerate().map(|(row, &depth)| {
        (trail[at + depth] + u32::from(row == case.wrong)) % vocab
    }).collect();
    DraftTree::plain(tokens, case.parents.to_vec())
}

struct Script { case: Case, trail: Vec<u32>, base: usize, vocab: u32, commits: usize, wrong: usize }
impl DraftProvider for Script {
    fn block_size(&self) -> usize { self.case.block }
    fn draft(&mut self, _: &mut dyn Model, start: usize, anchor: u32) -> Result<Vec<u32>, String> {
        let at = start.checked_sub(self.base).ok_or("small tree draft position")?;
        if self.trail.get(at) != Some(&anchor) { return Err("small tree anchor differs".into()); }
        self.trail.get(at+1..at+self.case.block).map(|x| x.to_vec()).ok_or_else(|| "small tree draft trail exhausted".into())
    }
    fn tree_proposal(&mut self, start: usize, _: u32, _: &[u32], _: &[u32]) -> Result<Option<DraftTree>, String> {
        Ok(Some(tree(self.case, &self.trail, start-self.base, self.vocab)))
    }
    fn commit_tree(&mut self, _: usize, _: &[u32], path: &[i32], _: u32) -> Result<(), String> {
        self.commits += 1;
        if !self.case.path.starts_with(path) { self.wrong += 1; }
        Ok(())
    }
}

// Captures contain canonical live full/window ranges, excluding ring slack and
// rejected packed tree rows. Report a first differing owner for useful failures.
fn live_equal(a: &imparo_kv::KvState, b: &imparo_kv::KvState) -> bool {
    if a.boundary != b.boundary || a.recurrent != b.recurrent { return false; }
    for (label,x,y) in [("full",&a.full,&b.full),("window",&a.window,&b.window)] {
        if x.len()!=y.len() { return false; }
        for (p,q) in x.iter().zip(y) {
            if p.layer!=q.layer || p.base_pos!=q.base_pos || p.positions!=q.positions || p.k!=q.k || p.v!=q.v {
                eprintln!("cuda-tree-small live mismatch kind={label} layer={} base={} positions={} k_equal={} v_equal={}",p.layer,p.base_pos,p.positions,p.k==q.k,p.v==q.v);
                return false;
            }
        }
    }
    true
}

pub fn run(model: &mut dyn Model, prompt: &[u32]) -> Result<(), String> {
    let case = match model.plan().config.architecture.as_str() {
        "lfm2" => LFM2,
        "gemma4" => {
            if model.kv_runtime().capacity!=1024 { return Err("E4B tree gate requires IMPARO_KV_CAP=1024".into()); }
            if std::env::var("IMPARO_LAB_E4B_TREE_DELAYED").as_deref() == Ok("1") { E4B_DELAYED } else { E4B }
        }
        name => return Err(format!("small tree fixture unsupported for {name}")),
    };
    let accepted=case.path.len();
    let probe_only=std::env::var_os("IMPARO_LAB_E4B_TREE_GRAPH_PROBE_DIR").is_some();
    if probe_only && model.plan().config.architecture!="gemma4" {
        return Err("Graph probe requires E4B".into());
    }
    let emit=if probe_only { 12 } else { EMIT };
    let base=prompt.len();
    let vocab=model.plan().config.vocab_size;
    if vocab<2 || base==0 || base+emit+9>model.kv_runtime().capacity { return Err("small tree reference exceeds context".into()); }
    let logits=model.forward(prompt,0)?;
    let mut next=imparo_cpu::ops::argmax_f32(&logits);
    let mut trail=vec![next];
    let mut reference_state=None;
    let prefix_state=model.kv_spill().ok_or("small tree prefix state unavailable")?;
    let mut stop_reference=None;
    let mut cursor_reference=None;
    let mut probe_states=Vec::new();
    for i in 0..emit+9 {
        next=model.forward_next(next,base+i)?;
        trail.push(next);
        if i+1==accepted { reference_state=model.kv_spill(); }
        if i==0 { stop_reference=model.kv_spill(); }
        if i+1==emit-1 { cursor_reference=model.kv_spill(); }
        if probe_only && (i+1)%accepted==0 && i+1<=4*accepted {
            probe_states.push(model.kv_spill().ok_or("Graph probe sequential KV unavailable")?);
        }
    }
    let reference_state=reference_state.ok_or("small tree reference state unavailable")?;
    let logits=model.forward(prompt,0)?;
    if imparo_cpu::ops::argmax_f32(&logits)!=trail[0] { return Err("small tree reset changed anchor".into()); }
    if !live_equal(&model.kv_spill().ok_or("small tree reset state unavailable")?,&prefix_state) { return Err("small tree reset changed live state".into()); }
    if probe_only {
        let mut captures=0;
        let mut replays=0;
        for (step, reference) in probe_states.iter().enumerate() {
            let at=step*accepted;
            let proposal=tree(case,&trail,at,vocab);
            if !model.prepare_greedy_tree(&proposal,base+at)? { return Err("Graph probe admission declined".into()); }
            let result=model.verify_greedy_tree(&proposal,base+at,accepted,&[])?;
            if result.path!=case.path || result.next_token!=trail[at+accepted]
                || model.kv_runtime().filled!=base+at+accepted
                || !live_equal(&model.kv_spill().ok_or("Graph probe live KV unavailable")?,reference) {
                return Err(format!("Graph probe output/KV mismatch at {}",base+at));
            }
            captures+=usize::from(result.submission==imparo_model::speculative::TreeSubmission::Capture);
            replays+=usize::from(result.submission==imparo_model::speculative::TreeSubmission::Replayed);
            println!("cuda-tree-small graph-probe start={} submission={:?} path={:?} live_kv_equal=true",base+at,result.submission,result.path);
        }
        let graph=std::env::var("IMPARO_LAB_E4B_TREE_GRAPH").as_deref()==Ok("1");
        if probe_states.len()!=4 || (graph && (captures==0 || replays<2)) || (!graph && (captures!=0 || replays!=0)) {
            return Err(format!("Graph probe did not exercise requested route capture={captures} replay={replays}"));
        }
        println!("cuda-tree-small graph-probe checks=4 captures={captures} replays={replays} verdict=PASS");
        return Ok(());
    }
    let proposal=tree(case,&trail,0,vocab);
    if !model.prepare_greedy_tree(&proposal,base)? { return Err("small tree admission declined".into()); }
    let result=model.verify_greedy_tree(&proposal,base,accepted,&[])?;
    let state=model.kv_spill().ok_or("small tree committed state unavailable")?;
    let recurrent_equal=state.recurrent==reference_state.recurrent;
    let kv_equal=live_equal(&state,&reference_state);
    let pass=result.path==case.path && result.next_token==trail[accepted]
        && model.kv_runtime().filled==base+accepted && kv_equal && recurrent_equal;
    println!("cuda-tree-small branch path={:?} kv_equal={kv_equal} recurrent_equal={recurrent_equal} verdict={}",result.path,if pass {"PASS"} else {"FAIL"});
    if !pass { return Err("small tree branch/state mismatch".into()); }
    next=result.next_token;
    for i in accepted..20 {
        next=model.forward_next(next,base+i)?;
        if next!=trail[i+1] { return Err(format!("small tree post-commit mismatch at {i}")); }
    }
    // A stop at the first prediction accepts only the root, discarding both branches.
    model.forward(prompt,0)?;
    if !model.prepare_greedy_tree(&proposal,base)? { return Err("small tree stop admission declined".into()); }
    let stop=model.verify_greedy_tree(&proposal,base,accepted,&[trail[1]])?;
    if stop.path!=[0] || stop.next_token!=trail[1] || model.kv_runtime().filled!=base+1 {
        return Err("small tree stop commit mismatch".into());
    }
    if !live_equal(&model.kv_spill().ok_or("small tree stop state unavailable")?,&stop_reference.ok_or("small tree stop reference unavailable")?) { return Err("small tree stop live state mismatch".into()); }
    if model.forward_next(stop.next_token,base+1)?!=trail[2] { return Err("small tree stop state mismatch".into()); }
    println!("cuda-tree-small stop root_only=true post_equal=true verdict=PASS");
    // Full cursor includes ordinary fallback at cell boundaries and a fresh reset.
    model.forward(prompt,0)?;
    let mut provider=Script { case,trail:trail.clone(),base,vocab,commits:0,wrong:0 };
    let result=imparo_model::speculative::continue_greedy(model,&mut provider,base,trail[0],EMIT,&[])?;
    let equal=result.tokens==trail[..EMIT];
    println!("cuda-tree-small cursor tokens={} equal={equal} commits={} wrong_paths={} sequential={} verdict={}",result.tokens.len(),provider.commits,provider.wrong,result.sequential_steps,
        if equal && provider.commits>0 && provider.wrong==0 {"PASS"} else {"FAIL"});
    if !equal || provider.commits==0 || provider.wrong!=0 { return Err("small tree full cursor mismatch".into()); }
    if model.kv_runtime().filled!=base+EMIT-1 || !live_equal(&model.kv_spill().ok_or("small tree cursor state unavailable")?,&cursor_reference.ok_or("small tree cursor reference unavailable")?) { return Err("small tree cursor live state mismatch".into()); }
    let reset_logits=model.forward(prompt,0)?;
    if imparo_cpu::ops::argmax_f32(&reset_logits)!=trail[0] || !live_equal(&model.kv_spill().ok_or("small tree final reset unavailable")?,&prefix_state) { return Err("small tree final reset mismatch".into()); }
    println!("cuda-tree-small cursor live_kv_equal=true final_reset=true verdict=PASS");
    Ok(())
}
