//! Real-weight routed-plan integration, anchored to the independent standalone lab.
use crate::{CudaBackend,context::CudaContext,ffi::*};
use imparo_backend::{Backend,BackendKnobs,BufId,Epilogue};
fn write(id:BufId,x:&[f32]){crate::correctness::write_f32_checked(id,0,x).unwrap();}
fn read(id:BufId,n:usize)->Vec<f32>{let mut x=vec![0.;n];crate::correctness::read_f32_checked(id,0,&mut x).unwrap();x}

#[test]
#[ignore = "requires idle SM86 and IMPARO_TEST_GATEUP_MMQ_FIXTURE; owns CUDA context"]
fn gpu_moe_gateup_mmq() {
    const NI:usize=2048; const NO:usize=1792; const NE:usize=32;
    const NT:usize=128; const ROWS:usize=512; const GUARD:usize=16;
    const STRIDE:u64=(NI/32*18*NO) as u64; const POISON:f32=1234567.;
    let dir=std::path::PathBuf::from(std::env::var("IMPARO_TEST_GATEUP_MMQ_FIXTURE").unwrap());
    let floats=|name:&str|->Vec<f32>{std::fs::read(dir.join("micro-001").join(name)).unwrap().chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect()};
    let input=floats("input.f32.bin"); let fixture_perm=floats("perm.u32.bin"); let fixture_seg=floats("seg.u32.bin");
    assert_eq!(input.len(),NT*NI);assert_eq!(fixture_perm.len(),ROWS);assert_eq!(fixture_seg.len(),NE+1);
    let mut route:Vec<Vec<u32>>=vec![vec![];NT];
    let mut lookup=vec![usize::MAX;NE*NT];
    for e in 0..NE { for row in fixture_seg[e].to_bits() as usize..fixture_seg[e+1].to_bits() as usize {
        let t=fixture_perm[row].to_bits() as usize; assert!(t<NT); route[t].push(e as u32);lookup[e*NT+t]=row;
    }}
    assert!(route.iter().all(|v|v.len()==4));
    let top:Vec<f32>=route.iter().flatten().map(|v|f32::from_bits(*v)).collect();
    let mut weights=std::fs::read(dir.join("gate.q4.bin")).unwrap();let up_off=weights.len() as u64;
    assert_eq!(up_off,STRIDE*NE as u64);weights.extend(std::fs::read(dir.join("up.q4.bin")).unwrap());
    let weights=Box::leak(weights.into_boxed_slice());
    unsafe{CudaContext::get().init_weights(weights.as_ptr(),weights.len() as u64,&[])}.unwrap();
    let be=CudaBackend;be.set_activation(Epilogue::Silu);
    let knob=be.knob_registry().iter().find(|k|k.name=="moe_gateup_mmq").unwrap();
    let pair=be.knob_registry().iter().find(|k|k.name=="moe_grouped_pair").unwrap();
    struct Restore{apply:fn(u32),old:u32,pair:fn(u32),pair_old:u32}
    impl Drop for Restore{fn drop(&mut self){(self.apply)(self.old);(self.pair)(self.pair_old);let _=crate::gateup_api::apply_at_boundary();CudaBackend.set_activation(Epilogue::Gelu);}}
    let _restore=Restore{apply:knob.apply,old:(knob.current)(),pair:pair.apply,pair_old:(pair.current)()};
    (pair.apply)(1);
    assert_eq!(unsafe{imparo_cuda_arena(16<<20)},0);
    let mut offset=0_u64;let mut perm_offset=0;
    for (id,n) in [(BufId::Cur,NT*NI),(BufId::G,ROWS*NO+GUARD),(BufId::U,ROWS*NI+GUARD),
        (BufId::O,NT*NO),(BufId::Model2,NT*NE),(BufId::Model4,ROWS),(BufId::Model5,ROWS),
        (BufId::Model6,ROWS),(BufId::Model7,NE+1),(BufId::Model8,ROWS)] {
        if id==BufId::Model5{perm_offset=offset;}
        assert_eq!(unsafe{imparo_cuda_place(id as u32,offset,(n*4) as u64)},0);offset=((offset+(n*4) as u64+255)/256)*256;
    }
    write(BufId::Cur,&input);write(BufId::Model4,&top);write(BufId::Model2,&vec![1.;NT*NE]);
    let plan=||assert!(be.moe_plan(BufId::Model4,BufId::Model2,BufId::Model5,BufId::Model6,BufId::Model7,BufId::Model8,NT as u32,NE as u32,4,true,1.));
    let call=|scratch:BufId|unsafe{crate::gateup_api::imparo_cuda_moe_gateup_mmq_pair_v1(1,0,up_off,STRIDE,BufId::Cur as u32,BufId::G as u32,scratch as u32,BufId::Model5 as u32,BufId::Model7 as u32,NI as u32,NO as u32,NE as u32,NT as u32,ROWS as u32)};
    let check=|file:&str| {
        let got=read(BufId::G,ROWS*NO+GUARD);let oracle=floats(file);
        let perm=read(BufId::Model5,ROWS);let seg=read(BufId::Model7,NE+1);
        assert_eq!(seg[NE].to_bits(),ROWS as u32);
        for e in 0..NE {for row in seg[e].to_bits() as usize..seg[e+1].to_bits() as usize {
            let token=perm[row].to_bits() as usize;assert!(token<NT);let oldrow=lookup[e*NT+token];assert_ne!(oldrow,usize::MAX);
            for n in 0..NO{assert_eq!(got[row*NO+n].to_bits(),oracle[GUARD+oldrow*NO+n].to_bits(),"native vs independent lab: e={e} token={token} channel={n}");}
        }}
        assert!(got[ROWS*NO..].iter().all(|v|*v==POISON));
        assert!(read(BufId::U,ROWS*NI+GUARD)[ROWS*NI..].iter().all(|v|*v==POISON));
    };
    for enabled in [0,1] {
        (knob.apply)(enabled);crate::gateup_api::apply_at_boundary().unwrap();
        write(BufId::G,&vec![POISON;ROWS*NO+GUARD]);write(BufId::U,&vec![POISON;ROWS*NI+GUARD]);
        be.begin_forward(false);plan();
        assert!(be.moe_grouped_pair_with_scratch(1,0,up_off,STRIDE,BufId::Cur,BufId::G,BufId::U,BufId::Model5,BufId::Model7,NI as u32,NO as u32,NE as u32,NT as u32,ROWS as u32,true));
        check(if enabled==0{"baseline.f32.bin"}else{"candidate.f32.bin"});be.end().unwrap();
    }
    assert_eq!(read(BufId::Cur,input.len()),input);
    // Dense publishes a cache which the routed Q8 overwrite must revoke.
    be.begin_forward(false);be.matmat(1,0,NI as u32,NO as u32,BufId::Cur,BufId::O,NT as u32);let dense=read(BufId::O,NT*NO);
    plan();assert_eq!(call(BufId::U),0);assert_eq!(call(BufId::U),-70,"route proof consumed exactly once");
    be.matmat(1,0,NI as u32,NO as u32,BufId::Cur,BufId::O,NT as u32);assert_eq!(read(BufId::O,NT*NO),dense,"stale shared Q8 cache");be.end().unwrap();
    // Foreign plans, changed epochs and physical aliases refuse before writes.
    be.begin_forward(false);assert_eq!(call(BufId::U),-70,"cross-forward proof");plan();
    assert_eq!(unsafe{imparo_cuda_place(BufId::Model0 as u32,0,(ROWS*NI*4) as u64)},0);
    assert_eq!(call(BufId::Model0),-70,"capacity-valid scratch/source physical alias");
    let current=read(BufId::Model5,ROWS);write(BufId::Model5,&current);assert_eq!(call(BufId::U),-70,"PERM epoch invalidation");
    assert_eq!(unsafe{imparo_cuda_place(BufId::Model0 as u32,perm_offset,(ROWS*4) as u64)},0);plan();
    write(BufId::Model0,&current);let before=read(BufId::G,ROWS*NO+GUARD);
    assert_eq!(call(BufId::U),-70,"alias write invalidation");assert_eq!(read(BufId::G,ROWS*NO+GUARD),before);
    // Invalid picks leave unwritten rows; producer clearing keeps every index bounded.
    let mut invalid=top.clone();invalid[0]=f32::from_bits(u32::MAX);write(BufId::Model4,&invalid);
    write(BufId::Model5,&vec![f32::from_bits(u32::MAX);ROWS]);plan();
    let p=read(BufId::Model5,ROWS);assert!(p.iter().all(|v|v.to_bits()<NT as u32));
    assert_eq!(read(BufId::Model7,NE+1)[NE].to_bits(),ROWS as u32-1);be.end().unwrap();
    for invalid in [0,2] {
        be.begin_forward(false);
        assert_ne!(unsafe{crate::gateup_api::imparo_cuda_moe_gateup_mmq_v1(invalid)},0);
        assert!(be.end().is_err(),"rejected policy changes must propagate pending errors");
    }
    println!("GateUp: full native vs independent lab, actual plan permutation, U reuse, guards, cache, owner, epochs, physical alias and invalid-pick tail pass");
}
