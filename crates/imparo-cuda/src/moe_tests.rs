//! Fixed routed-FFN proof: CPU f64 oracle, both row spaces, bias, ties and refusal.
use crate::{CudaBackend, context::CudaContext, ffi::*};
use imparo_backend::{Backend, BackendKnobs, BufId, Epilogue, ExpertGating};

fn write(id: BufId, x: &[f32]) { crate::correctness::write_f32_checked(id,0,x).unwrap(); }
fn read(id: BufId,n:usize)->Vec<f32> {
    let mut x=vec![0.;n];crate::correctness::read_f32_checked(id,0,&mut x).unwrap();x
}
fn near(a:f64,b:f64,what:&str) { assert!((a-b).abs()<2e-5*(1.+b.abs()),"{what}: {a} vs {b}"); }

#[test]
#[ignore = "requires idle SM86 and IMPARO_TEST_DOWN_MMQ_FIXTURE; owns CUDA context"]
fn gpu_moe_down_mmq() {
    const NI: usize = 1792; const NO: usize = 2048; const NE: usize = 32;
    const ROWS: usize = 512; const GUARD: usize = 16;
    const STRIDE: u64 = (NI / 32 * 18 * NO) as u64;
    const POISON: f32 = 1234567.;
    let dir=std::path::PathBuf::from(std::env::var("IMPARO_TEST_DOWN_MMQ_FIXTURE").unwrap());
    let weights=Box::leak(std::fs::read(dir.join("down.q4.bin")).unwrap().into_boxed_slice());
    assert_eq!(weights.len(),STRIDE as usize*NE);
    let floats=|path:std::path::PathBuf| -> Vec<f32> {std::fs::read(path).unwrap().chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect()};
    let input=floats(dir.join("micro-001/input.f32.bin"));
    let seg=floats(dir.join("micro-001/seg.u32.bin"));
    assert_eq!(input.len(),ROWS*NI);assert_eq!(seg.len(),NE+1);
    unsafe {CudaContext::get().init_weights(weights.as_ptr(),weights.len() as u64,&[])}.unwrap();
    let be=CudaBackend;
    let knob=be.knob_registry().iter().find(|k|k.name=="moe_down_mmq").unwrap();
    struct Restore{apply:fn(u32),old:u32}
    impl Drop for Restore{fn drop(&mut self){(self.apply)(self.old);let _=crate::down_api::apply_at_boundary();}}
    let _restore=Restore{apply:knob.apply,old:(knob.current)()};
    for (id,n) in [(BufId::Cur,ROWS*NI),(BufId::U,ROWS*NI),(BufId::G,ROWS*NO+GUARD),(BufId::O,ROWS*NO+GUARD),(BufId::Model5,ROWS),(BufId::Model7,NE+1)] {
        assert_eq!(unsafe{imparo_cuda_alloc(id as u32,(n*4) as u64)},0);
    }
    write(BufId::Cur,&input);write(BufId::U,&input);write(BufId::Model7,&seg);
    write(BufId::Model5,&vec![f32::from_bits(0);ROWS]);
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let run=|nt:u32,rows:u32|assert!(be.moe_grouped(1,0,STRIDE,BufId::Cur,BufId::O,BufId::Model5,BufId::Model7,NI as u32,NO as u32,NE as u32,nt,rows,true,nt>1));
    for selected in [0,1] {
        (knob.apply)(selected);crate::down_api::apply_at_boundary().unwrap();
        write(BufId::O,&vec![POISON;ROWS*NO+GUARD]);
        be.begin_forward(false);
        assert_eq!(unsafe{crate::down_api::imparo_cuda_moe_down_mmq_v1(selected)},0);
        assert_ne!(unsafe{crate::down_api::imparo_cuda_moe_down_mmq_v1(1-selected)},0);
        assert_ne!(unsafe{crate::down_api::imparo_cuda_moe_down_mmq_v1(2)},0);
        run(128,ROWS as u32);be.end().unwrap();
        let got=read(BufId::O,ROWS*NO+GUARD);
        let file=if selected==0 {"baseline.f32.bin"}else{"candidate.f32.bin"};
        let oracle=floats(dir.join("micro-001").join(file));
        assert_eq!(oracle.len(),ROWS*NO+2*GUARD);
        assert_eq!(bits(&got),bits(&oracle[GUARD..]),"native adapter vs independently checked standalone");
    }
    // Dense Q4 publishes the shared Q8 cache. MoE must invalidate it before
    // replacing its contents with group-major 512 work rows.
    let dense=||{be.begin_forward(false);be.matmat(1,0,NI as u32,NO as u32,BufId::U,BufId::G,128);be.end().unwrap();read(BufId::G,128*NO)};
    let before=dense();be.begin_forward(false);run(128,ROWS as u32);be.end().unwrap();
    assert_eq!(bits(&dense()),bits(&before),"MoE scratch must not leave a stale dense Q8 cache");
    assert_eq!(bits(&read(BufId::Cur,input.len())),bits(&input));
    // M1 is outside the candidate domain and keeps its established arithmetic.
    let tiny_seg:Vec<f32>=(0..=NE).map(|i|f32::from_bits(if i==0 {0}else{4})).collect();
    write(BufId::Model7,&tiny_seg);let mut prior=None;
    for selected in [0,1] {
        (knob.apply)(selected);crate::down_api::apply_at_boundary().unwrap();
        write(BufId::O,&vec![POISON;ROWS*NO+GUARD]);be.begin_forward(true);run(1,4);be.end().unwrap();
        let got=read(BufId::O,ROWS*NO+GUARD);assert!(got[4*NO..].iter().all(|v|*v==POISON));
        let current=bits(&got);if let Some(ref p)=prior {assert_eq!(p,&current,"M1 fallback");}prior=Some(current);
    }
}

#[test]
#[ignore = "requires an idle CUDA GPU and IMPARO_TEST_ROUTER_FIXTURE; owns context"]
fn gpu_moe_router_f32() {
    const NI:usize=2048; const NO:usize=32; const GUARD:usize=16;
    const QNI:usize=2560; const MIB:u64=1024*1024;
    let dir=std::path::PathBuf::from(std::env::var("IMPARO_TEST_ROUTER_FIXTURE").unwrap());
    let mut bytes=std::fs::read(dir.join("router.f32.bin")).unwrap();
    assert_eq!(bytes.len(),NI*NO*4);
    let wf:Vec<f32>=bytes.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
    // Synthetic F32 gate only exercises the existing Qwen provider's owner grow.
    let qoff=bytes.len() as u64;
    let qweights:Vec<f32>=(0..QNI*NO).map(|i|((i*37+11)%257) as f32/4096.-128./4096.).collect();
    for value in &qweights {bytes.extend(value.to_le_bytes());}
    let weights=Box::leak(bytes.into_boxed_slice());
    let input=std::fs::read(dir.join("input.f32.bin")).unwrap();
    let x:Vec<f32>=input.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
    assert_eq!(x.len(),NI);
    unsafe {CudaContext::get().init_weights(weights.as_ptr(),weights.len() as u64,&[])}.unwrap();
    // Choose before execution; Qwen freezes policy 4 on its first eligible call.
    // This isolated GPU test intentionally does not mutate that frozen policy.
    assert_eq!(unsafe {imparo_cuda_set_ptq_prefill_tensorcore(4)},0);
    let be=CudaBackend;
    let knob=be.knob_registry().iter().find(|k| k.name=="moe_router_f32").unwrap();
    struct Restore {apply:fn(u32),old:u32}
    impl Drop for Restore {fn drop(&mut self){(self.apply)(self.old);let _=crate::router_api::apply_at_boundary();}}
    let _restore=Restore{apply:knob.apply,old:(knob.current)()};
    assert_eq!(unsafe {imparo_cuda_alloc(BufId::Cur as u32,(3*NI*4) as u64)},0);
    assert_eq!(unsafe {imparo_cuda_alloc(BufId::G as u32,((2*NO+GUARD)*4) as u64)},0);
    assert_eq!(unsafe {imparo_cuda_alloc(BufId::U as u32,((QNI+GUARD)*4) as u64)},0);
    let mut inputs=vec![0.25;NI];inputs.extend(&x);inputs.extend(x.iter().map(|v| -*v));
    write(BufId::Cur,&inputs);
    let poison=1234567.0_f32;
    let allocated=|| {
        let mut bytes=0_u64;
        assert_eq!(unsafe {imparo_cuda_memory_info(std::ptr::null_mut(),std::ptr::null_mut(),&mut bytes)},0);
        bytes
    };
    let mut moe_reference=Vec::new();
    for nt in [1_usize,2] {
        let mut arms=Vec::new();
        let mut before_router=0_u64;
        for selected in [0,1] {
            (knob.apply)(selected);
            crate::router_api::apply_at_boundary().unwrap();
            write(BufId::G,&vec![poison;2*NO+GUARD]);
            be.begin_forward(nt==1);
            assert_eq!(unsafe {crate::router_api::imparo_cuda_moe_router_f32_v1(selected)},0);
            assert_ne!(unsafe {crate::router_api::imparo_cuda_moe_router_f32_v1(1-selected)},0);
            assert_ne!(unsafe {crate::router_api::imparo_cuda_moe_router_f32_v1(2)},0);
            be.matmat_from(0,0,NI as u32,NO as u32,BufId::Cur,BufId::G,nt as u32,1);
            be.end().unwrap();
            let output=read(BufId::G,2*NO+GUARD);
            assert!(output[nt*NO..].iter().all(|v|*v==poison));
            for t in 0..nt {for r in 0..NO {
                let expected:f64=(0..NI).map(|i|f64::from(wf[r*NI+i])*f64::from(inputs[(t+1)*NI+i])).sum();
                assert!(output[t*NO+r].is_finite());near(f64::from(output[t*NO+r]),expected,"router FP64");
            }}
            if nt==1 {
                if selected==0 {before_router=allocated();}
                else {
                    assert_eq!(allocated(),before_router+4*MIB,"MoE F32 needs only its 4 MiB workspace");
                    moe_reference=output.clone();
                }
            }
            arms.push(output);
        }
        let top4=|out:&[f32]| {let mut ids:Vec<usize>=(0..NO).collect();ids.sort_by(|&a,&b|out[b].total_cmp(&out[a]).then(a.cmp(&b)));ids[..4].to_vec()};
        assert_eq!(top4(&arms[0]),top4(&arms[1]));
        if nt==2 {assert_eq!(arms[0],arms[1],"M2 must retain original arithmetic");}
    }
    assert_eq!(read(BufId::Cur,3*NI),inputs);

    // Same execution owner: MoE workspace -> full existing Qwen owner -> MoE.
    let mut qinput:Vec<f32>=(0..QNI).map(|i|((i*19+7)%127) as f32/128.-63./128.).collect();
    qinput.extend(vec![poison;GUARD]);
    write(BufId::U,&qinput);
    write(BufId::G,&vec![poison;2*NO+GUARD]);
    let before_grow=allocated();
    be.begin_forward(true);
    be.matmat(0,qoff,QNI as u32,NO as u32,BufId::U,BufId::G,1);
    be.end().unwrap();
    let qoutput=read(BufId::G,2*NO+GUARD);
    assert_eq!(allocated(),before_grow+81*MIB,"same owner must grow from 4 to 85 MiB");
    assert_eq!(unsafe {imparo_cuda_ptq_prefill_tensorcore()},4);
    assert!(qoutput[NO..].iter().all(|v|*v==poison));
    for r in 0..NO {
        let expected:f64=(0..QNI).map(|i|f64::from(qweights[r*QNI+i])*f64::from(qinput[i])).sum();
        assert!(qoutput[r].is_finite());near(f64::from(qoutput[r]),expected,"Qwen F32 after owner grow FP64");
    }
    assert_eq!(read(BufId::U,QNI+GUARD),qinput);
    let after_grow=allocated();
    write(BufId::G,&vec![poison;2*NO+GUARD]);
    be.begin_forward(true);
    be.matmat_from(0,0,NI as u32,NO as u32,BufId::Cur,BufId::G,1,1);
    be.end().unwrap();
    assert_eq!(read(BufId::G,2*NO+GUARD).iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
        moe_reference.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"MoE output and guards must remain bitwise identical after owner grow");
    assert_eq!(allocated(),after_grow,"returning to MoE reuses the grown owner");
    assert_eq!(read(BufId::Cur,3*NI),inputs);
}

#[test]
#[ignore = "requires an idle CUDA GPU; owns process-global context"]
fn gpu_moe_routed_ffn() {
    const NI:usize=256; const NO:usize=32; const NE:usize=8; const K:usize=3;
    let mut bytes=vec![0_u8;256];
    let bias:Vec<f32>=(0..NE).map(|e|if e==6 {0.6}else{0.}).collect();
    for (e,b) in bias.iter().enumerate() {bytes[e*4..e*4+4].copy_from_slice(&b.to_le_bytes());}
    let mut stacks=Vec::new();
    for kind in [0_u32,1,2] {
        let off=bytes.len(); let mut vals=Vec::new();
        for e in 0..NE {for r in 0..NO {for block in 0..NI/32 {
            let q:Vec<i32>=(0..32).map(|i|((e*11+r*7+block*3+i)%15) as i32-7).collect();
            vals.extend(q.iter().map(|v|f64::from(*v)/32.));
            match kind {
                0=>for v in q {bytes.extend((v as f32/32.).to_le_bytes());},
                1=> {bytes.extend(0x2800_u16.to_le_bytes());for i in 0..16 {bytes.push(((q[i]+8)|((q[i+16]+8)<<4)) as u8);}},
                2=> {bytes.extend(0x2800_u16.to_le_bytes());bytes.extend(q.iter().map(|v|*v as i8 as u8));},
                _=>unreachable!(),
            }
        }}}
        stacks.push((kind,off as u64,((bytes.len()-off)/NE) as u64,vals));
    }
    let q6off=bytes.len();let mut q6vals=Vec::new();
    for r in 0..NE*NO {
        let mut block=vec![0_u8;210];
        for (i,v) in block[..192].iter_mut().enumerate(){*v=((i*37+r*11)%256) as u8;}
        for (i,v) in block[192..208].iter_mut().enumerate(){*v=((i*3+r)%11) as i8 as u8;*v=(*v as i8-5) as u8;}
        block[208..210].copy_from_slice(&0x1800_u16.to_le_bytes());
        let mut vals=vec![0_f32;NI];imparo_cpu::quants::row_codec(14).unwrap()(&block,&mut vals);
        q6vals.extend(vals.iter().map(|v|f64::from(*v)));bytes.extend(block);
    }
    stacks.push((7,q6off as u64,(NO*210) as u64,q6vals));
    let weights=Box::leak(bytes.into_boxed_slice());
    unsafe {CudaContext::get().init_weights(weights.as_ptr(),weights.len() as u64,&[])}.unwrap();
    let be=CudaBackend;
    for id in [BufId::Model1,BufId::Model2,BufId::Model3,BufId::Model4,BufId::Model5,
        BufId::Model6,BufId::Model7,BufId::Model8,BufId::Cur,BufId::G,BufId::U,BufId::O,BufId::Tokens] {
        assert_eq!(unsafe {imparo_cuda_alloc(id as u32,65536)},0);
    }
    assert!(be.supports_moe());assert!(be.supports_top_k_rows(8));assert!(!be.supports_top_k_rows(9));
    let q6=&stacks[3];
    be.row(7,q6.1,NI as u32,11,0.5,BufId::Cur,3);
    let row=read(BufId::Cur,NI+3);
    for i in 0..NI {near(row[i+3] as f64,q6.3[11*NI+i]*0.5,"Q6 embedding");}
    be.write_u32(BufId::Tokens,0,&[2,7,11]);
    assert!(be.gather_rows(7,q6.1,NI as u32,(NE*NO) as u32,1.,BufId::Cur,0,BufId::Tokens,3));
    let rows=read(BufId::Cur,3*NI);
    for (t,r) in [2,7,11].iter().enumerate(){for i in 0..NI{near(rows[t*NI+i] as f64,q6.3[r*NI+i],"Q6 gathered embedding");}}
    be.matmat(7,q6.1,NI as u32,NO as u32,BufId::Cur,BufId::G,3);
    let proj=read(BufId::G,3*NO);
    for t in 0..3 {for r in 0..NO {
        near(proj[t*NO+r] as f64,(0..NI).map(|i|f64::from(rows[t*NI+i])*q6.3[r*NI+i]).sum(),"Q6 projection");
    }}
    // Stable ties and negative infinity, independent of probabilities.
    write(BufId::Model3,&[f32::NEG_INFINITY,2.,2.,-1.,2.,f32::NEG_INFINITY]);
    assert!(be.top_k_rows(BufId::Model3,BufId::Model4,6,1,6));
    assert_eq!(read(BufId::Model4,6).iter().map(|x|x.to_bits()).collect::<Vec<_>>(),[1,2,4,3,0,5]);
    for nt in [1_usize,11] {for gating in [ExpertGating::Softmax,ExpertGating::Sigmoid] {
        let scores:Vec<f32>=(0..nt*NE).map(|i|((i*7%19) as f32-9.)/4.).collect();
        let x:Vec<f32>=(0..nt*NI).map(|i|((i*13%29) as f32-14.)/16.).collect();
        write(BufId::Model1,&scores);write(BufId::Cur,&x);
        assert!(be.moe_gate(BufId::Model1,BufId::Model2,BufId::Model3,0,nt as u32,NE as u32,gating));
        let probs=read(BufId::Model2,nt*NE);let sel=read(BufId::Model3,nt*NE);
        for t in 0..nt {
            let row=&scores[t*NE..(t+1)*NE];let max=row.iter().copied().fold(f32::NEG_INFINITY,f32::max);
            let denom: f64=row.iter().map(|v|f64::from(v-max).exp()).sum();
            for e in 0..NE {
                let p=match gating {ExpertGating::Softmax=>f64::from(row[e]-max).exp()/denom,ExpertGating::Sigmoid=>1./(1.+(-f64::from(row[e])).exp())};
                near(probs[t*NE+e] as f64,p,"gate");near(sel[t*NE+e] as f64,p+f64::from(bias[e]),"bias");
            }
        }
        assert!(be.top_k_rows(BufId::Model3,BufId::Model4,NE as u32,nt as u32,K as u32));
        let picks:Vec<usize>=read(BufId::Model4,nt*K).iter().map(|x|x.to_bits() as usize).collect();
        for t in 0..nt {let mut ids:Vec<usize>=(0..NE).collect();ids.sort_by(|&a,&b|sel[t*NE+b].total_cmp(&sel[t*NE+a]).then(a.cmp(&b)));assert_eq!(&picks[t*K..(t+1)*K],&ids[..K]);}
        for norm in [false,true] {
            assert!(be.moe_plan(BufId::Model4,BufId::Model2,BufId::Model5,BufId::Model6,BufId::Model7,BufId::Model8,nt as u32,NE as u32,K as u32,norm,0.75));
            let perm:Vec<usize>=read(BufId::Model5,nt*K).iter().map(|v|v.to_bits() as usize).collect();
            let seg:Vec<usize>=read(BufId::Model7,NE+1).iter().map(|v|v.to_bits() as usize).collect();
            let inv:Vec<usize>=read(BufId::Model8,nt*K).iter().map(|v|v.to_bits() as usize).collect();
            let w=read(BufId::Model6,nt*K);
            assert_eq!(seg[0],0);assert_eq!(seg[NE],nt*K);
            let mut sorted=inv.clone();sorted.sort_unstable();assert_eq!(sorted,(0..nt*K).collect::<Vec<_>>());
            for t in 0..nt {for j in 0..K {
                let e=picks[t*K+j];let row=inv[t*K+j];assert_eq!(perm[row],t);assert!(row>=seg[e] && row<seg[e+1]);
                let denom=if norm {(0..K).map(|z|probs[t*NE+picks[t*K+z]]).sum::<f32>().max(6.103_515_6e-5)}else{1.};
                near(w[row] as f64,(probs[t*NE+e]/denom*0.75) as f64,"routing weight");
            }}
            for (kind,off,stride,vals) in &stacks {
                let mut outputs=Vec::new();
                for work in [false,true] {
                    if work {
                        let mut gathered=vec![0.;nt*K*NI];
                        for row in 0..nt*K {gathered[row*NI..(row+1)*NI].copy_from_slice(&x[perm[row]*NI..(perm[row]+1)*NI]);}
                        write(BufId::U,&gathered);
                    }
                    assert!(be.moe_grouped(*kind,*off,*stride,if work{BufId::U}else{BufId::Cur},BufId::G,BufId::Model5,BufId::Model7,NI as u32,NO as u32,NE as u32,nt as u32,(nt*K) as u32,work,false));
                    let y=read(BufId::G,nt*K*NO);
                    for e in 0..NE {for row in seg[e]..seg[e+1] {for r in 0..NO {
                        let expected:f64=(0..NI).map(|i|vals[(e*NO+r)*NI+i]*f64::from(x[perm[row]*NI+i])).sum();
                        near(y[row*NO+r] as f64,expected,"grouped FP64");
                    }}}
                    assert!(be.moe_combine(BufId::G,BufId::Model6,BufId::Model8,BufId::O,NO as u32,K as u32,nt as u32));
                    let out=read(BufId::O,nt*NO);
                    for t in 0..nt {for r in 0..NO {
                        let expected:f64=(0..K).map(|j|{let row=inv[t*K+j];f64::from(w[row])*f64::from(y[row*NO+r])}).sum();
                        near(out[t*NO+r] as f64,expected,"combine");
                    }}
                    outputs.push(out);
                }
                assert_eq!(outputs[0],outputs[1],"token/work input spaces must agree");
            }
        }
    }}
    let before=read(BufId::O,32);
    assert!(!be.moe_combine(BufId::G,BufId::Model6,BufId::Model8,BufId::O,32,9,1));
    assert!(!be.moe_grouped(3,0,128,BufId::Cur,BufId::O,BufId::Model5,BufId::Model7,64,32,8,1,3,false,false));
    assert!(!be.moe_gate(BufId::Model1,BufId::Model1,BufId::Model3,u64::MAX,1,8,ExpertGating::Sigmoid));
    assert_eq!(read(BufId::O,32),before,"refused calls must not write");
    be.end().unwrap();
    println!("MoE gate/top-k/plan/grouped/combine: 1/11 tokens, F32/Q4/Q8/Q6_K, FP64 oracle, both row spaces and refusal pass; Q6 embedding/head agree with CPU codec");
}

#[test]
#[ignore = "requires an idle CUDA GPU; owns process-global context"]
fn gpu_moe_grouped_pair_q4() {
    grouped_pair_q4_case(false);
}

#[test]
#[ignore = "requires an idle CUDA GPU; owns process-global context"]
fn gpu_moe_active_experts_q4() {
    grouped_pair_q4_case(true);
}

fn grouped_pair_q4_case(active_test: bool) {
    const NI: usize = 2048;
    const NO: usize = 1792;
    const NE: usize = 32;
    const K: usize = 4;
    const GUARD: usize = 16;
    const POISON: f32 = f32::from_bits(0x4f12_3456);
    const RB: usize = NI / 32 * 18;
    const STRIDE: usize = NO * RB;

    // Canonical Q4_0, independent gate/up stacks and varying exact f16 scales.
    // Keep the bytes as the CPU reference input; do not reuse native decoding.
    let mut bytes = Vec::with_capacity(2 * NE * STRIDE);
    for plane in 0..2 {
        for e in 0..NE {
            for r in 0..NO {
                for block in 0..NI / 32 {
                    let scale = [0x2000_u16, 0x2400, 0x2800][(e + r + block + plane) % 3];
                    bytes.extend(scale.to_le_bytes());
                    let q = |i: usize| ((e * 11 + r * 7 + block * 3 + i * 5 + plane * (13 + e)) % 15) as u8 + 1;
                    for i in 0..16 { bytes.push(q(i) | (q(i + 16) << 4)); }
                }
            }
        }
    }
    let weights = Box::leak(bytes.into_boxed_slice());
    unsafe { CudaContext::get().init_weights(weights.as_ptr(), weights.len() as u64, &[]) }.unwrap();
    let be = CudaBackend;
    let knob = be.knob_registry().iter().find(|k| k.name == crate::knobs::MOE_GROUPED_PAIR_KNOB).unwrap();
    let active_knob = be.knob_registry().iter().find(|k| k.name == "moe_active_experts").unwrap();
    struct Restore { apply: fn(u32), previous: u32, active_apply: fn(u32), active_previous: u32 }
    impl Drop for Restore {
        fn drop(&mut self) {
            (self.apply)(self.previous);
            (self.active_apply)(self.active_previous);
            let _ = crate::active_api::apply_at_boundary();
            CudaBackend.set_activation(Epilogue::Gelu);
        }
    }
    let _restore = Restore { apply: knob.apply, previous: (knob.current)(), active_apply: active_knob.apply, active_previous: (active_knob.current)() };
    let active_policy = |enabled: bool| {
        (active_knob.apply)(u32::from(enabled));
        crate::active_api::apply_at_boundary().unwrap();
    };
    be.set_activation(Epilogue::Silu);
    let capacity = (128 * K * NO + GUARD) as u64 * 4;
    for id in [BufId::Model1, BufId::Model2, BufId::Model3, BufId::Model4,
        BufId::Model5, BufId::Model6, BufId::Model7, BufId::Model8,
        BufId::Cur, BufId::G, BufId::U, BufId::O] {
        assert_eq!(unsafe { imparo_cuda_alloc(id as u32, capacity) }, 0);
    }
    // Deliberately insufficient output buffer for a no-write refusal below.
    assert_eq!(unsafe { imparo_cuda_alloc(BufId::Tokens as u32, 4) }, 0);
    let up_off = (NE * STRIDE) as u64;
    let codec = imparo_cpu::quants::row_codec(2).unwrap();

    for nt in [1_usize, 128] {
        active_policy(false); // all-expert reference, including the ordinary pair gate
        let rows = nt * K;
        let count = rows * NO;
        let prefill_chunk = nt > 1;
        let scores: Vec<f32> = (0..nt).flat_map(|t| (0..NE).map(move |e| {
            if active_test && nt == 1 && [1_usize, 9, 18, 29].contains(&e) { 16. + e as f32 / 32. }
            else { ((e + NE - t % NE) % NE) as f32 / 4. }
        })).collect();
        let x: Vec<f32> = (0..nt).flat_map(|t| (0..NI).map(move |i| ((t * 17 + i * 13 + (i / 32) * (t + 1)) % 61) as f32 / 64. - 0.46875)).collect();
        write(BufId::Model1, &scores);
        write(BufId::Cur, &x);
        assert!(be.moe_gate(BufId::Model1, BufId::Model2, BufId::Model3, u64::MAX, nt as u32, NE as u32, ExpertGating::Softmax));
        assert!(be.top_k_rows(BufId::Model3, BufId::Model4, NE as u32, nt as u32, K as u32));
        assert!(be.moe_plan(BufId::Model4, BufId::Model2, BufId::Model5, BufId::Model6, BufId::Model7, BufId::Model8, nt as u32, NE as u32, K as u32, true, 1.));
        let perm: Vec<usize> = read(BufId::Model5, rows).iter().map(|v| v.to_bits() as usize).collect();
        let seg: Vec<usize> = read(BufId::Model7, NE + 1).iter().map(|v| v.to_bits() as usize).collect();
        assert_eq!(seg[0], 0);
        assert_eq!(seg[NE], rows);
        assert!(perm.iter().all(|&t| t < nt));
        if nt == 128 { assert!(seg.windows(2).all(|s| s[0] < s[1])); }

        let poison = vec![POISON; count + GUARD];
        write(BufId::G, &poison);
        write(BufId::U, &poison);
        assert!(be.moe_grouped(1, 0, STRIDE as u64, BufId::Cur, BufId::G, BufId::Model5, BufId::Model7, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32, false, prefill_chunk));
        assert!(be.moe_grouped(1, up_off, STRIDE as u64, BufId::Cur, BufId::U, BufId::Model5, BufId::Model7, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32, false, prefill_chunk));
        be.act_mul(BufId::G, BufId::U, count as u32);
        let reference = read(BufId::G, count + GUARD);
        assert!(reference[..count].iter().all(|v| v.is_finite() && *v != POISON));
        assert_eq!(&reference[count..], &poison[count..]);

        let pair = |kind, gate, up, stride, src, dst, ni, no, ne, n_tok, n_rows| {
            be.moe_grouped_pair(kind, gate, up, stride, src, dst, BufId::Model5, BufId::Model7, ni, no, ne, n_tok, n_rows, prefill_chunk)
        };
        write(BufId::O, &poison);
        (knob.apply)(0);
        assert!(!pair(1, 0, up_off, STRIDE as u64, BufId::Cur, BufId::O, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32));
        assert_eq!(read(BufId::O, count + GUARD), poison, "disabled pair must not write");
        (knob.apply)(1);
        if active_test {
            active_policy(true);
            if nt == 1 {
                // M1 has one token, so work-row order is fixed. For nt128 keep
                // the reference plan: atomic work-row order may change on a
                // second plan, while this test compares raw, uncombined rows.
                assert!(be.moe_plan(BufId::Model4, BufId::Model2, BufId::Model5, BufId::Model6, BufId::Model7, BufId::Model8, nt as u32, NE as u32, K as u32, true, 1.));
                let tail: Vec<u32> = read(BufId::Model7, 2 * NE + 2).iter().map(|v| v.to_bits()).collect();
                assert_eq!(tail[NE + 1], K as u32);
                assert_eq!(&tail[NE + 2..NE + 2 + K], &[1, 9, 18, 29]);
            }
        }
        assert!(pair(1, 0, up_off, STRIDE as u64, BufId::Cur, BufId::O, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32));
        let actual = read(BufId::O, count + GUARD);
        for (i, (got, want)) in actual.iter().zip(&reference).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "nt={nt}, output/guard index={i}");
        }

        // Independent CPU Q4 reader + f64 dot/SiLU on five boundary/interior
        // dimensions of every routed row; full outputs above remain bitwise.
        let mut gate_row = vec![0.; NI];
        let mut up_row = vec![0.; NI];
        for e in 0..NE {
            for r in [0, 1, 31, 32, NO - 1] {
                let off = e * STRIDE + r * RB;
                codec(&weights[off..off + RB], &mut gate_row);
                let off = up_off as usize + off;
                codec(&weights[off..off + RB], &mut up_row);
                for row in seg[e]..seg[e + 1] {
                    let token = &x[perm[row] * NI..(perm[row] + 1) * NI];
                    let gate: f64 = gate_row.iter().zip(token).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
                    let up: f64 = up_row.iter().zip(token).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
                    near(f64::from(actual[row * NO + r]), gate / (1. + (-gate).exp()) * up, "paired Q4 independent FP64");
                }
            }
        }

        if active_test && nt == 1 {
            let refresh = || {
                assert!(be.moe_plan(BufId::Model4, BufId::Model2, BufId::Model5, BufId::Model6, BufId::Model7, BufId::Model8, 1, NE as u32, K as u32, true, 1.));
            };
            let check_pair = |seg_id: BufId| {
                write(BufId::O, &poison);
                assert!(be.moe_grouped_pair(1, 0, up_off, STRIDE as u64, BufId::Cur, BufId::O, BufId::Model5, seg_id, NI as u32, NO as u32, NE as u32, 1, K as u32, false));
                assert_eq!(read(BufId::O, count + GUARD).iter().map(|v| v.to_bits()).collect::<Vec<_>>(), reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            };
            // The old route ABI must also produce a valid compact list. Its
            // gate/top-k/plan arithmetic and all grouped input spaces stay intact.
            assert_eq!(unsafe { crate::route_api::imparo_cuda_moe_route_v1(
                BufId::Model1 as u32, BufId::Model2 as u32, BufId::Model3 as u32,
                BufId::Model4 as u32, BufId::Model5 as u32, BufId::Model6 as u32,
                BufId::Model7 as u32, BufId::Model8 as u32, u64::MAX, 1, NE as u32, K as u32, 0, 1, 1.) }, 0);
            check_pair(BufId::Model7);

            // The same canonical byte stack is a legal 1792 -> 2048 down matrix
            // (equal per-expert byte size). Compare all work-row outputs and a
            // separate CPU decoder/f64 reference for its boundary dimensions.
            let down_count = K * NI;
            let down = || assert!(be.moe_grouped(1, 0, STRIDE as u64, BufId::O, BufId::U,
                BufId::Model5, BufId::Model7, NO as u32, NI as u32, NE as u32, 1, K as u32, true, false));
            active_policy(false);
            write(BufId::U, &vec![POISON; down_count + GUARD]);
            down();
            let down_reference = read(BufId::U, down_count + GUARD);
            active_policy(true);
            refresh();
            write(BufId::U, &vec![POISON; down_count + GUARD]);
            down();
            let down_actual = read(BufId::U, down_count + GUARD);
            assert_eq!(down_actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), down_reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            let down_rb = NO / 32 * 18;
            let mut decoded = vec![0_f32; NO];
            for e in [1_usize, 9, 18, 29] {
                for r in [0_usize, NI - 1] {
                    let off = e * STRIDE + r * down_rb;
                    codec(&weights[off..off + down_rb], &mut decoded);
                    for row in seg[e]..seg[e + 1] {
                        let expected: f64 = decoded.iter().zip(&actual[row * NO..(row + 1) * NO]).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
                        near(f64::from(down_actual[row * NI + r]), expected, "active down FP64");
                    }
                }
            }

            // An overwritten tail has a new SEG epoch. Leave the valid offsets
            // but poison the compact count: fallback must still write every row.
            let mut stale = read(BufId::Model7, 2 * NE + 2);
            stale[NE + 1] = f32::from_bits(0);
            write(BufId::Model7, &stale);
            check_pair(BufId::Model7);
            refresh();
            // Invalid control and a value change during a forward must not mutate
            // any routing data or silently retain a different requested policy.
            let before = read(BufId::Model7, 2 * NE + 2);
            assert_ne!(unsafe { crate::active_api::imparo_cuda_moe_active_experts_v1(2) }, 0);
            be.begin_forward(true);
            assert_eq!(unsafe { crate::active_api::imparo_cuda_moe_active_experts_v1(1) }, 0);
            assert_ne!(unsafe { crate::active_api::imparo_cuda_moe_active_experts_v1(0) }, 0);
            assert_eq!(read(BufId::Model7, 2 * NE + 2), before);
            be.end().unwrap();
            check_pair(BufId::Model7);

            // The original n_expert+1 SEG allocation is still accepted by the
            // unchanged plan/pair APIs and takes the original full expert grid.
            assert_eq!(unsafe { imparo_cuda_alloc(BufId::Tokens as u32, ((NE + 1) * 4) as u64) }, 0);
            assert!(be.moe_plan(BufId::Model4, BufId::Model2, BufId::Model5, BufId::Model6,
                BufId::Tokens, BufId::Model8, 1, NE as u32, K as u32, true, 1.));
            check_pair(BufId::Tokens);

            // An arena alias can change only the tail while SEG's own epoch is
            // unchanged. The owner invalidates the stamp by physical overlap.
            assert_eq!(unsafe { imparo_cuda_arena(1024) }, 0);
            assert_eq!(unsafe { imparo_cuda_place(BufId::Model7 as u32, 0, ((2 * NE + 2) * 4) as u64) }, 0);
            assert_eq!(unsafe { imparo_cuda_place(BufId::Tmp as u32, ((NE + 1) * 4) as u64, (K * 4) as u64) }, 0);
            refresh();
            write(BufId::Tmp, &[f32::from_bits(0)]);
            check_pair(BufId::Model7);
            // A producer output aliasing the tail must never acquire a stamp.
            assert!(be.moe_plan(BufId::Model4, BufId::Model2, BufId::Model5, BufId::Model6,
                BufId::Model7, BufId::Tmp, 1, NE as u32, K as u32, true, 1.));
            check_pair(BufId::Model7);
        }

        write(BufId::O, &poison);
        be.set_activation(Epilogue::Gelu);
        assert!(!pair(1, 0, up_off, STRIDE as u64, BufId::Cur, BufId::O, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32));
        be.set_activation(Epilogue::Silu);
        for (kind, gate, up, stride, ni, no, ne, n_tok, n_rows) in [
            (0, 0, up_off, STRIDE as u64, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32),
            (2, 0, up_off, STRIDE as u64, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32),
            (1, 1, up_off, STRIDE as u64, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32),
            (1, 0, u64::MAX - 1, STRIDE as u64, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32),
            (1, 0, up_off, STRIDE as u64 - 2, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32),
            (1, 0, up_off, u64::MAX - 1, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32),
            (1, 0, up_off, STRIDE as u64, NI as u32 - 1, NO as u32, NE as u32, nt as u32, rows as u32),
            (1, 0, up_off, STRIDE as u64, NI as u32, 0, NE as u32, nt as u32, rows as u32),
            (1, 0, up_off, STRIDE as u64, NI as u32, NO as u32, 257, nt as u32, rows as u32),
            (1, 0, up_off, STRIDE as u64, NI as u32, NO as u32, NE as u32, 0, rows as u32),
            (1, 0, up_off, STRIDE as u64, NI as u32, NO as u32, NE as u32, nt as u32, nt as u32 * 9),
        ] {
            assert!(!pair(kind, gate, up, stride, BufId::Cur, BufId::O, ni, no, ne, n_tok, n_rows));
        }
        assert_eq!(read(BufId::O, count + GUARD), poison, "refused shapes/weights/activation must not write");
        assert!(!pair(1, 0, up_off, STRIDE as u64, BufId::O, BufId::O, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32));
        assert_eq!(read(BufId::O, count + GUARD), poison, "refused alias must not write");
        write(BufId::Tokens, &[POISON]);
        assert!(!pair(1, 0, up_off, STRIDE as u64, BufId::Cur, BufId::Tokens, NI as u32, NO as u32, NE as u32, nt as u32, rows as u32));
        assert_eq!(read(BufId::Tokens, 1), [POISON], "refused capacity must not write");
        assert_eq!(read(BufId::Cur, x.len()), x, "pair must preserve token input");
    }
    be.end().unwrap();
    println!("MoE grouped pair Q4: NE32/K4 NI2048/NO1792 nt1/128 full bitwise against grouped+grouped+SiLU, sampled independent FP64, poison guards, disabled/unsupported/alias/capacity refusal pass");
    if active_test {
        println!("MoE active experts: non-contiguous [1,9,18,29], pair/down all-expert bitwise, down FP64, old route, nt128 fallback, stale SEG epoch, arena alias, short SEG and forward-boundary refusal pass; Graph replay and cross-owner isolation are static-review coverage only");
    }
}

#[test]
#[ignore = "requires an idle CUDA GPU; owns process-global context"]
fn gpu_moe_route_one() {
    const NE: usize = 32;
    const K: usize = 4;
    const GUARD: usize = 8;
    const POISON: f32 = f32::from_bits(0x4f12_3456);
    let bias: Vec<f32> = (0..NE).map(|e| match e { 31 => 0.75, 7 => -0.125, _ => 0. }).collect();
    let mut bytes = vec![0_u8; 256];
    for (e, value) in bias.iter().enumerate() {
        bytes[e * 4..e * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    let weights = Box::leak(bytes.into_boxed_slice());
    unsafe { CudaContext::get().init_weights(weights.as_ptr(), weights.len() as u64, &[]) }.unwrap();
    let be = CudaBackend;
    let knob = be.knob_registry().iter().find(|k| k.name == crate::knobs::MOE_ROUTE_KNOB).unwrap();
    struct Restore { apply: fn(u32), previous: u32 }
    impl Drop for Restore { fn drop(&mut self) { (self.apply)(self.previous); } }
    let _restore = Restore { apply: knob.apply, previous: (knob.current)() };
    let ids = [BufId::Model1, BufId::Model2, BufId::Model3, BufId::Model4,
        BufId::Model5, BufId::Model6, BufId::Model7, BufId::Model8];
    for id in ids { assert_eq!(unsafe { imparo_cuda_alloc(id as u32, 65536) }, 0); }
    assert_eq!(unsafe { imparo_cuda_alloc(BufId::Tokens as u32, 4) }, 0);
    let outputs = [(ids[1], NE), (ids[2], NE), (ids[3], 2 * K), (ids[4], K),
        (ids[5], K), (ids[6], NE + 1), (ids[7], K)];
    let poison_outputs = || { for (id, count) in outputs { write(id, &vec![POISON; count + GUARD]); } };
    let snapshots = || -> Vec<Vec<f32>> { outputs.iter().map(|&(id, count)| read(id, count + GUARD)).collect() };
    let route = |bias_off, nt, gating, normalise, scale| {
        be.moe_route(ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
            bias_off, nt, NE as u32, K as u32, gating, normalise, scale)
    };

    // Fixed semantic cases, not a tuning sweep. Repeated finite scores exercise
    // lower-id ties; the low sigmoid case exercises the normalisation floor.
    for pattern in 0_usize..3 {
        let scores: Vec<f32> = (0..NE).map(|e| match pattern {
            0 => ((e * 7 % 19) as f32 - 9.) / 4.,
            1 => 0.,
            _ => -100.,
        }).collect();
        let mut scores_guarded = scores.clone();
        scores_guarded.extend([POISON; GUARD]);
        write(ids[0], &scores_guarded);
        for gating in [ExpertGating::Softmax, ExpertGating::Sigmoid] {
            for with_bias in [false, true] {
                let bias_off = if with_bias { 0 } else { u64::MAX };
                for normalise in [false, true] {
                    let scale = if pattern == 2 { -0.5 } else if normalise { 0.75 } else { 1.25 };
                    poison_outputs();
                    assert!(be.moe_gate(ids[0], ids[1], ids[2], bias_off, 1, NE as u32, gating));
                    assert!(be.top_k_rows(ids[2], ids[3], NE as u32, 1, K as u32));
                    assert!(be.moe_plan(ids[3], ids[1], ids[4], ids[5], ids[6], ids[7], 1, NE as u32, K as u32, normalise, scale));
                    let reference = snapshots();
                    for (out, &(_, count)) in reference.iter().zip(&outputs) {
                        assert_eq!(&out[count..], &[POISON; GUARD]);
                    }
                    poison_outputs();
                    (knob.apply)(0);
                    assert!(!route(bias_off, 1, gating, normalise, scale));
                    assert!(snapshots().iter().flatten().all(|v| v.to_bits() == POISON.to_bits()));
                    (knob.apply)(1);
                    assert!(route(bias_off, 1, gating, normalise, scale));
                    let actual = snapshots();
                    for (out_index, (got, want)) in actual.iter().zip(&reference).enumerate() {
                        for (index, (a, b)) in got.iter().zip(want).enumerate() {
                            assert_eq!(a.to_bits(), b.to_bits(), "route pattern={pattern}, bias={with_bias}, norm={normalise}, output={out_index}, index={index}");
                        }
                    }
                    assert_eq!(read(ids[0], scores_guarded.len()), scores_guarded);

                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let denominator: f64 = scores.iter().map(|&s| f64::from(s - max).exp()).sum();
                    let probabilities: Vec<f64> = scores.iter().map(|&s| match gating {
                        ExpertGating::Softmax => f64::from(s - max).exp() / denominator,
                        ExpertGating::Sigmoid => 1. / (1. + (-f64::from(s)).exp()),
                    }).collect();
                    let selection: Vec<f64> = probabilities.iter().enumerate().map(|(e, &p)| p + if with_bias { f64::from(bias[e]) } else { 0. }).collect();
                    for e in 0..NE {
                        near(f64::from(actual[0][e]), probabilities[e], "route FP64 probability");
                        near(f64::from(actual[1][e]), selection[e], "route FP64 selection");
                    }
                    let mut picks: Vec<usize> = (0..NE).collect();
                    picks.sort_by(|&a, &b| selection[b].total_cmp(&selection[a]).then(a.cmp(&b)));
                    picks.truncate(K);
                    assert_eq!(actual[2][..K].iter().map(|v| v.to_bits() as usize).collect::<Vec<_>>(), picks);
                    let denom = if normalise { picks.iter().map(|&e| probabilities[e]).sum::<f64>().max(6.103_515_625e-5) } else { 1. };
                    for e in 0..=NE {
                        assert_eq!(actual[5][e].to_bits() as usize, picks.iter().filter(|&&p| p < e).count());
                    }
                    for (j, &e) in picks.iter().enumerate() {
                        let row = picks.iter().filter(|&&p| p < e).count();
                        assert_eq!(actual[6][j].to_bits() as usize, row);
                        assert_eq!(actual[3][row].to_bits(), 0);
                        near(f64::from(actual[2][K + j]), selection[e], "route top-k values");
                        near(f64::from(actual[4][row]), probabilities[e] / denom * f64::from(scale), "route unbiased routing weight");
                    }
                }
            }
        }
    }

    // Inspect every owned word on refusal, including the input and small-buffer
    // substitute. The native admission is checked directly as well as via the
    // Backend guard, so nt=128 cannot be hidden by its early Rust fallback.
    const REFUSAL_WORDS: usize = 16384;
    let poison = vec![POISON; REFUSAL_WORDS];
    for id in ids { write(id, &poison); }
    write(BufId::Tokens, &[POISON]);
    let check_unchanged = || {
        for id in ids { assert_eq!(read(id, REFUSAL_WORDS), poison, "refused route changed {id:?}"); }
        assert_eq!(read(BufId::Tokens, 1), [POISON]);
    };
    let raw = |buffers: [u32; 8], bias, nt, ne, k, gating, norm, scale| unsafe {
        crate::route_api::imparo_cuda_moe_route_v1(buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5], buffers[6], buffers[7], bias, nt, ne, k, gating, norm, scale) == 0
    };
    let buffers = ids.map(|id| id as u32);
    assert!(!route(u64::MAX, 128, ExpertGating::Softmax, true, 1.));
    for (bias, nt, ne, k, gating, norm, scale) in [
        (u64::MAX, 0, NE as u32, K as u32, 0, 1, 1.),
        (u64::MAX, 128, NE as u32, K as u32, 0, 1, 1.),
        (u64::MAX, 1, 0, K as u32, 0, 1, 1.),
        (u64::MAX, 1, 257, K as u32, 0, 1, 1.),
        (u64::MAX, 1, NE as u32, 0, 0, 1, 1.),
        (u64::MAX, 1, NE as u32, 9, 0, 1, 1.),
        (u64::MAX, 1, 2, K as u32, 0, 1, 1.),
        (u64::MAX, 1, NE as u32, K as u32, 2, 1, 1.),
        (u64::MAX, 1, NE as u32, K as u32, 0, 2, 1.),
        (u64::MAX, 1, NE as u32, K as u32, 0, 1, f32::NAN),
        (u64::MAX, 1, NE as u32, K as u32, 0, 1, f32::INFINITY),
        (1, 1, NE as u32, K as u32, 0, 1, 1.),
        (u64::MAX - 3, 1, NE as u32, K as u32, 0, 1, 1.),
        (weights.len() as u64 - 4, 1, NE as u32, K as u32, 0, 1, 1.),
    ] {
        assert!(!raw(buffers, bias, nt, ne, k, gating, norm, scale));
    }
    check_unchanged();
    for position in 0..buffers.len() {
        let mut small = buffers;
        small[position] = BufId::Tokens as u32;
        assert!(!raw(small, u64::MAX, 1, NE as u32, K as u32, 0, 1, 1.));
        let mut invalid = buffers;
        invalid[position] = u32::MAX;
        assert!(!raw(invalid, u64::MAX, 1, NE as u32, K as u32, 0, 1, 1.));
    }
    for a in 0..buffers.len() { for b in a + 1..buffers.len() {
        let mut alias = buffers;
        alias[b] = alias[a];
        assert!(!raw(alias, u64::MAX, 1, NE as u32, K as u32, 0, 1, 1.));
    }}
    check_unchanged();
    be.end().unwrap();
    println!("MoE route nt1 NE32/K4: all seven outputs/top-k halves/guards bitwise against gate+stable-topk+plan; independent FP64 probability, selection and unbiased weight; ties, sigmoid floor, bias/none, norm/scale and no-write refusal pass");
}
