//! Run explicitly on an idle CUDA device, in a fresh process.
use crate::{CudaBackend, context::CudaContext, ffi::*};
use imparo_backend::{Backend, BufId, KvLayout};

#[test]
#[ignore = "requires an idle CUDA GPU and owns process-global native state"]
fn gpu_cobatch_slot_isolation_and_grow() {
    let weights = Box::leak(vec![0_u8; 4096].into_boxed_slice());
    unsafe {
        CudaContext::get().init_weights(weights.as_ptr(), weights.len() as u64, &[])
    }
    .unwrap();
    let be = CudaBackend;
    unsafe {
        imparo_cuda_set_kv_types(1, 1);
        for id in [BufId::Recur, BufId::RecurSnap, BufId::K] {
            assert_eq!(imparo_cuda_alloc(id as u32, 128), 0);
        }
    }
    let mut sizes = [8192_u64, 4096]; // width32 F16, full128 / ring64
    let mut layouts = [
        KvLayout {
            layer: 0,
            logical_slots: 128,
            k_stride: 64,
            v_stride: 64,
            ..KvLayout::default()
        },
        KvLayout {
            layer: 1,
            ..KvLayout::default()
        },
    ];
    unsafe {
        assert_eq!(
            imparo_cuda_alloc_kv_layout(2, sizes.as_ptr(), layouts.as_ptr(), 2),
            0
        );
    }
    be.end().unwrap();
    let read = |id| {
        let mut values = [0_f32; 32];
        crate::correctness::read_f32_checked(id, 0, &mut values).unwrap();
        values
    };
    let ring_read = || {
        let mut bytes = [0_u8; 64];
        unsafe {
            assert_eq!(imparo_cuda_read_kv(1, 0, 0, bytes.as_mut_ptr(), 64), 0);
        }
        bytes
    };
    crate::correctness::write_f32_checked(BufId::Recur, 0, &[3.0; 32]).unwrap();
    crate::correctness::write_f32_checked(BufId::RecurSnap, 0, &[5.0; 32]).unwrap();
    assert!(be.set_slots(3, &[1]));
    assert!(!be.set_slots(3, &[1, 1]));
    assert!(!be.select_slot(3));
    unsafe {
        assert_eq!(imparo_cuda_set_kv_pages(0, [1_u32, 0].as_ptr(), 2), 0);
    }
    crate::correctness::write_f32_checked(BufId::K, 0, &[1.0; 32]).unwrap();
    be.kv_store(BufId::K, 0, 32, 0, 1, false, 0);
    be.kv_store(BufId::K, 1, 32, 0, 1, false, 63);
    be.end().unwrap();
    let ring0 = ring_read();
    assert!(be.select_slot(1));
    assert_eq!(read(BufId::Recur), [0.0; 32]);
    assert_eq!(read(BufId::RecurSnap), [0.0; 32]);
    assert_eq!(ring_read(), [0; 64]);
    crate::correctness::write_f32_checked(BufId::Recur, 0, &[7.0; 32]).unwrap();
    crate::correctness::write_f32_checked(BufId::RecurSnap, 0, &[11.0; 32]).unwrap();
    unsafe {
        assert_eq!(imparo_cuda_set_kv_pages(0, [0_u32, 1].as_ptr(), 2), 0);
    }
    crate::correctness::write_f32_checked(BufId::K, 0, &[2.0; 32]).unwrap();
    be.kv_store(BufId::K, 0, 32, 0, 1, false, 0);
    be.kv_store(BufId::K, 1, 32, 0, 1, false, 63);
    be.end().unwrap();
    let ring1 = ring_read();
    assert_ne!(ring0, ring1);
    assert!(!be.release_slot(1));
    assert!(!be.release_slot(2)); // lazy, never allocated
    // Grow while a private ring is selected and slot zero is inactive.
    sizes[0] = 16384;
    layouts[0].logical_slots = 256;
    unsafe {
        assert_eq!(
            imparo_cuda_grow_kv_layout(2, sizes.as_ptr(), layouts.as_ptr(), 2),
            0
        );
    }
    assert_eq!(ring_read(), ring1);
    assert!(be.select_slot(0));
    assert_eq!(ring_read(), ring0);
    assert_eq!(read(BufId::Recur), [3.0; 32]);
    assert_eq!(read(BufId::RecurSnap), [5.0; 32]);
    // A store after grow still follows slot zero's non-identity page table.
    crate::correctness::write_f32_checked(BufId::K, 0, &[4.0; 32]).unwrap();
    be.kv_store(BufId::K, 0, 32, 1, 1, false, 0);
    be.end().unwrap();
    let mut at_mapped = [0_u8; 64];
    let mut at_identity = [0_u8; 64];
    unsafe {
        assert_eq!(
            imparo_cuda_read_kv(0, 0, 65 * 64, at_mapped.as_mut_ptr(), 64),
            0
        );
        assert_eq!(
            imparo_cuda_read_kv(0, 0, 64, at_identity.as_mut_ptr(), 64),
            0
        );
    }
    assert!(at_mapped.chunks_exact(2).all(|x| x == [0, 0x44])); // F16 4
    assert_eq!(at_identity, [0; 64]);
    assert!(be.select_slot(1));
    assert_eq!(read(BufId::Recur), [7.0; 32]);
    assert_eq!(read(BufId::RecurSnap), [11.0; 32]);
    assert!(be.release_slot(0)); // borrowed ring + standalone recurrent planes
    assert!(be.select_slot(0));
    assert_eq!(read(BufId::Recur), [0.0; 32]);
    assert_eq!(read(BufId::RecurSnap), [0.0; 32]);
    assert_eq!(ring_read(), [0; 64]);
    assert!(be.release_slot(1));
    assert!(be.select_slot(1));
    assert_eq!(read(BufId::Recur), [0.0; 32]);
    assert_eq!(ring_read(), [0; 64]);
    eprintln!(
        "CoBatch GPU gate: slot isolation, both recurrent planes, private/borrowed rings, page mapping, grow, release/reuse passed"
    );
}

#[test]
#[ignore = "requires an idle CUDA GPU and owns process-global native state"]
fn gpu_cobatch_e4b_attention_rows() {
    let weights = Box::leak(vec![1.0_f32; 1024].into_boxed_slice());
    unsafe { CudaContext::get().init_weights(weights.as_ptr().cast(), 4096, &[]) }
        .unwrap();
    let be = CudaBackend;
    let hd = 256_u32;
    let heads = 4_u32;
    let width = heads * hd;
    unsafe {
        imparo_cuda_set_kv_types(1, 1);
        for (id, n) in [
            (BufId::Q, 2 * width),
            (BufId::Attn, 2 * width),
            (BufId::K, 6 * hd),
            (BufId::V, 6 * hd),
        ] {
            assert_eq!(imparo_cuda_alloc(id as u32, u64::from(n) * 4), 0);
        }
        let layout = KvLayout {
            layer: 0,
            logical_slots: 256,
            k_stride: u64::from(hd) * 2,
            v_stride: u64::from(hd) * 2,
            ..KvLayout::default()
        };
        assert_eq!(
            imparo_cuda_alloc_kv_layout(
                1,
                [256 * u64::from(hd) * 2].as_ptr(),
                &layout,
                1
            ),
            0
        );
    }
    be.end().unwrap();
    assert!(be.set_slots(2, &[]));
    let input: Vec<f32> = (0..2 * width)
        .map(|i| ((i * 13 % 97) as f32 - 48.0) / 49.0)
        .collect();
    let mut expected = Vec::new();
    let mut expected_q = Vec::new();
    for slot in 0..2_u32 {
        assert!(be.select_slot(slot));
        unsafe {
            assert_eq!(
                imparo_cuda_set_kv_pages(0, [slot, 2, 3, 1 - slot].as_ptr(), 4),
                0
            );
        }
        let values: Vec<f32> = (0..6 * hd)
            .map(|i| ((i * 7 + slot * 31) % 101) as f32 / 101.0)
            .collect();
        crate::correctness::write_f32_checked(BufId::K, 0, &values).unwrap();
        crate::correctness::write_f32_checked(BufId::V, 0, &values).unwrap();
        crate::correctness::write_f32_checked(
            BufId::Q,
            0,
            &input[(slot * width) as usize..((slot + 1) * width) as usize],
        )
        .unwrap();
        be.begin_forward(false);
        be.kv_store(BufId::K, 0, hd, 0, 6, false, 0);
        be.kv_store(BufId::V, 0, hd, 0, 6, true, 0);
        unsafe {
            imparo_cuda_head_norm_rope_hadamard(
                BufId::Q as u32,
                0,
                hd,
                1e-6,
                heads,
                slot * 5,
                1,
                hd,
                10000.0,
                core::ptr::null(),
                0,
            );
        }
        be.attention(0, hd, heads, 1, hd, slot * 5, 1.0, 0, 1, slot * 5 + 1, 0);
        be.end().unwrap();
        let mut out = vec![0.0; width as usize];
        let mut q = out.clone();
        crate::correctness::read_f32_checked(BufId::Attn, 0, &mut out).unwrap();
        crate::correctness::read_f32_checked(BufId::Q, 0, &mut q).unwrap();
        expected.extend(out);
        expected_q.extend(q);
    }
    crate::correctness::write_f32_checked(BufId::Q, 0, &input).unwrap();
    assert!(be.set_decode_rows(Some(imparo_backend::RowRoute::Fast)));
    be.begin_forward(false);
    assert!(be.head_norm_rope_at(
        BufId::Q,
        0,
        hd,
        1e-6,
        heads,
        &[0, 5],
        hd,
        10000.0,
        None
    ));
    let rows = [
        imparo_backend::SlotRow { slot: 0, pos: 0 },
        imparo_backend::SlotRow { slot: 1, pos: 5 },
    ];
    assert!(be.attention_slot_rows(0, hd, heads, 1, hd, 1.0, 0, &rows, &[1, 6], 0));
    be.end().unwrap();
    assert!(be.set_decode_rows(None));
    let mut got = vec![0.0; expected.len()];
    let mut q = got.clone();
    crate::correctness::read_f32_checked(BufId::Attn, 0, &mut got).unwrap();
    crate::correctness::read_f32_checked(BufId::Q, 0, &mut q).unwrap();
    let diff = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(got.iter().all(|x| x.is_finite()));
    assert_eq!(
        q, expected_q,
        "row positions must preserve one-row head arithmetic"
    );
    assert!(diff <= 1e-6, "attention row maximum difference {diff}");
    eprintln!(
        "E4B row gate: independent positions, mapped KV, head and attention; max difference={diff}"
    );
}

#[test]
#[ignore = "requires an idle CUDA GPU and owns process-global native state"]
fn gpu_cobatch_e4b_q4_rows() {
    let weights = Box::leak(vec![1.0_f32; 1024].into_boxed_slice());
    unsafe { CudaContext::get().init_weights(weights.as_ptr().cast(), 4096, &[]) }
        .unwrap();
    let be = CudaBackend;
    be.set_kv_types(2, 2);
    for (id, bytes) in [
        (BufId::Q, 32768),
        (BufId::Attn, 32768),
        (BufId::K, 1025 * 1024 * 4),
        (BufId::V, 1025 * 1024 * 4),
        (BufId::Kdq, 1025 * 1024 * 2),
        (BufId::Vdq, 1025 * 1024 * 2),
    ] {
        be.alloc(id, bytes).unwrap();
    }
    let sizes = [1024 * 512 / 32 * 18_u64, 256 * 1024 / 32 * 18];
    let layouts = [
        KvLayout {
            layer: 0,
            ..KvLayout::default()
        },
        KvLayout {
            layer: 1,
            logical_slots: 256,
            k_stride: 1024 / 32 * 18,
            v_stride: 1024 / 32 * 18,
            ..KvLayout::default()
        },
    ];
    be.alloc_kv_layout(&sizes, &layouts).unwrap();
    be.end().unwrap();
    assert!(be.set_slots(2, &[0]));
    for (layer, hd, heads, kvheads, positions, ring, window) in [
        (
            0_u32,
            256_u32,
            8_u32,
            2_u32,
            [1023_u32, 1024_u32],
            1023_u32,
            512_u32,
        ),
        (1, 512, 8, 2, [63, 64], 0, 0),
    ] {
        let width = heads * hd;
        let kvw = kvheads * hd;
        let stride = kvw / 32 * 18;
        let input = |n: u32, seed: u32| {
            (0..n)
                .map(|i| ((i * 13 + seed * 19) % 101) as f32 / 41.0 - 1.0)
                .collect::<Vec<_>>()
        };
        let q = input(2 * width, 1);
        let k = input(2 * kvw, 3);
        let v = input(2 * kvw, 7);
        let mut expected = Vec::new();
        let mut snapshots = Vec::new();
        for slot in 0..2_u32 {
            assert!(be.select_slot(slot));
            if ring == 0 {
                unsafe {
                    assert_eq!(
                        imparo_cuda_set_kv_pages(
                            layer,
                            [slot * 2, slot * 2 + 1, 2 - slot * 2, 3 - slot * 2]
                                .as_ptr(),
                            4
                        ),
                        0
                    );
                }
            }
            let seed_rows = if ring != 0 {
                1024
            } else {
                positions[slot as usize] + 1
            };
            for buf in [BufId::K, BufId::V] {
                crate::correctness::write_f32_checked(
                    buf,
                    0,
                    &input(seed_rows * kvw, slot + 5),
                )
                .unwrap();
                be.kv_store(buf, layer, kvw, 0, seed_rows, buf == BufId::V, ring);
            }
            be.end().unwrap();
            for (buf, data, n) in [
                (BufId::Q, &q, width),
                (BufId::K, &k, kvw),
                (BufId::V, &v, kvw),
            ] {
                crate::correctness::write_f32_checked(
                    buf,
                    0,
                    &data[(slot * n) as usize..((slot + 1) * n) as usize],
                )
                .unwrap();
            }
            be.begin_forward(true);
            be.head_norm_rope_hadamard(
                BufId::Q,
                0,
                hd,
                1e-6,
                heads,
                positions[slot as usize],
                1,
                hd,
                10000.0,
                None,
                32,
            );
            be.kv_head_postprocess(
                BufId::K,
                BufId::V,
                0,
                hd,
                1e-6,
                kvheads,
                positions[slot as usize],
                1,
                hd,
                10000.0,
                None,
                32,
                32,
            );
            be.kv_store(
                BufId::K,
                layer,
                kvw,
                positions[slot as usize],
                1,
                false,
                ring,
            );
            be.kv_store(
                BufId::V,
                layer,
                kvw,
                positions[slot as usize],
                1,
                true,
                ring,
            );
            be.attention(
                layer,
                hd,
                heads,
                kvheads,
                kvw,
                positions[slot as usize],
                1.0,
                window,
                1,
                positions[slot as usize] + 1,
                ring,
            );
            be.hadamard(BufId::Attn, width, 32);
            be.end().unwrap();
            let mut out = vec![0.0; width as usize];
            crate::correctness::read_f32_checked(BufId::Attn, 0, &mut out).unwrap();
            expected.extend(out);
            let mut pair = Vec::new();
            for is_v in [false, true] {
                let mut bytes = vec![0; sizes[layer as usize] as usize];
                be.read_kv_bytes(layer, is_v, 0, &mut bytes);
                pair.push(bytes);
                let pos = positions[slot as usize];
                let physical = if ring != 0 {
                    pos & ring
                } else {
                    (slot * 2 + pos / 64) * 64 + pos % 64
                };
                be.write_kv_bytes(
                    layer,
                    is_v,
                    u64::from(physical * stride),
                    &vec![0; stride as usize],
                );
            }
            snapshots.push(pair);
        }
        for (buf, data) in [(BufId::Q, &q), (BufId::K, &k), (BufId::V, &v)] {
            crate::correctness::write_f32_checked(buf, 0, data).unwrap();
        }
        assert!(be.set_decode_rows(Some(imparo_backend::RowRoute::Fast)));
        be.begin_forward(false);
        assert!(be.head_norm_rope_hadamard_at(
            BufId::Q,
            0,
            hd,
            1e-6,
            heads,
            &positions,
            hd,
            10000.0,
            None,
            32
        ));
        assert!(be.kv_head_postprocess_at(
            BufId::K,
            BufId::V,
            0,
            hd,
            1e-6,
            kvheads,
            &positions,
            hd,
            10000.0,
            None,
            32,
            32
        ));
        let rows = [
            imparo_backend::SlotRow {
                slot: 0,
                pos: positions[0],
            },
            imparo_backend::SlotRow {
                slot: 1,
                pos: positions[1],
            },
        ];
        let invalid = [
            rows[0],
            imparo_backend::SlotRow {
                slot: 99,
                pos: positions[1],
            },
        ];
        assert!(!be.kv_store_slot_rows(BufId::K, layer, kvw, &invalid, false, ring));
        assert!(be.kv_store_slot_rows(BufId::K, layer, kvw, &rows, false, ring));
        assert!(be.kv_store_slot_rows(BufId::V, layer, kvw, &rows, true, ring));
        assert!(be.attention_slot_rows(
            layer,
            hd,
            heads,
            kvheads,
            kvw,
            1.0,
            window,
            &rows,
            &[positions[0] + 1, positions[1] + 1],
            ring
        ));
        be.hadamard(BufId::Attn, 2 * width, 32);
        be.end().unwrap();
        assert!(be.set_decode_rows(None));
        let mut got = vec![0.0; expected.len()];
        crate::correctness::read_f32_checked(BufId::Attn, 0, &mut got).unwrap();
        let diff = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            got.iter().all(|x| x.is_finite()) && diff <= 1e-6,
            "Q4 row attention layer {layer} diff {diff}"
        );
        // Slot 1 is still selected: the row scope must restore bindings before this read.
        for slot in [1, 0] {
            assert!(be.select_slot(slot));
            for is_v in [false, true] {
                let mut bytes = vec![0; sizes[layer as usize] as usize];
                be.read_kv_bytes(layer, is_v, 0, &mut bytes);
                // Full-attention storage is shared: compare only this slot's
                // two physical pages. Ring storage is private, so compare all.
                let range = if ring == 0 {
                    let start = (slot * 128 * stride) as usize;
                    start..start + (128 * stride) as usize
                } else {
                    0..bytes.len()
                };
                let wanted = &snapshots[slot as usize][usize::from(is_v)];
                let first = range.clone().find(|&i| bytes[i] != wanted[i]);
                assert!(
                    first.is_none(),
                    "Q4 slot {slot} layer {layer} V={is_v} first bad byte {first:?}"
                );
            }
        }
        eprintln!(
            "E4B Q4 row gate: D{hd}, ring={ring}, independent positions={positions:?}, cache bytes equal, output maxdiff={diff}"
        );
    }
}

#[test]
#[ignore = "requires an idle Windows SM86 30-SM CUDA GPU and owns process-global native state"]
fn gpu_cobatch_e4b_q4_retained_long_attention() {
    const POS: u32 = 6144;
    const PREFILL_POS: u32 = POS - 1;
    const CAPACITY: u32 = 6656;
    const KV_HEADS: u32 = 2;
    let weights = Box::leak(vec![0_u8; 4096].into_boxed_slice());
    unsafe {
        CudaContext::get().init_weights(weights.as_ptr(), weights.len() as u64, &[])
    }
    .unwrap();
    let be = CudaBackend;
    be.set_kv_types(2, 2);
    unsafe {
        // Retained E4B attention choices, including D256 vector and D512 packed
        // M1. Default knobs miss the route used by the long-context failure.
        for (knob, value) in [
            (23, 1024),
            (29, 1),
            (30, 1),
            (31, 1),
            (32, 1),
            (33, 3),
            (34, 1),
            (35, 1),
            (36, 1),
            (37, 1),
        ] {
            imparo_cuda_set_knob(knob, value);
        }
        assert_eq!(imparo_cuda_set_e4b_retained_decode_policy(1), 0);
        assert_eq!(imparo_cuda_e4b_retained_decode_policy(), 1);
    }
    for (id, bytes) in [
        (BufId::Q, 2 * 8 * 512 * 4_u64),
        (BufId::Attn, 2 * 8 * 512 * 4),
        // Dequant writes physical rows. The mapped conversation lives beyond
        // one request's capacity, so scratch must cover the shared KV pool.
        (BufId::Kdq, u64::from(2 * CAPACITY) * 1024 * 2),
        (BufId::Vdq, u64::from(2 * CAPACITY) * 1024 * 2),
    ] {
        be.alloc(id, bytes).unwrap();
    }
    let full_stride = 1024 / 32 * 18_u64;
    be.alloc_kv_layout(
        &[1024 * 512 / 32 * 18, u64::from(2 * CAPACITY) * full_stride],
        &[
            KvLayout {
                layer: 0,
                ..KvLayout::default()
            },
            KvLayout {
                layer: 1,
                logical_slots: u64::from(2 * CAPACITY),
                k_stride: full_stride,
                v_stride: full_stride,
                ..KvLayout::default()
            },
        ],
    )
    .unwrap();
    be.end().unwrap();
    assert!(be.set_slots(2, &[0]));

    let identity: Vec<u32> = (0..2 * CAPACITY / 64).collect();
    let mapped: Vec<u32> = identity
        .iter()
        .map(|page| (page + CAPACITY / 64) % (2 * CAPACITY / 64))
        .collect();
    let set_pages = |pages: &[u32]| unsafe {
        assert_eq!(
            imparo_cuda_set_kv_pages(1, pages.as_ptr(), pages.len() as u32),
            0
        );
    };
    let raw_q4 = |rows: u32, width: u32, seed: u32| {
        let mut bytes = Vec::with_capacity((rows * width / 32 * 18) as usize);
        for block in 0..rows * width / 32 {
            // Exact half scales 1/64 and 1/32; varying packed lanes distinguish
            // rows, heads, keys, values and conversations without a quantizer.
            let scale = if (block + seed) % 3 == 0 {
                0x2800_u16
            } else {
                0x2400
            };
            bytes.extend_from_slice(&scale.to_le_bytes());
            for lane in 0..16 {
                let lo = (block * 7 + lane * 3 + seed * 5) % 16;
                let hi = (block * 11 + lane * 5 + seed * 7) % 16;
                bytes.push((lo | (hi << 4)) as u8);
            }
        }
        bytes
    };
    let read_output = |count: usize| {
        let mut out = vec![0.0; count];
        crate::correctness::read_f32_checked(BufId::Attn, 0, &mut out).unwrap();
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite attention output"
        );
        out
    };
    let difference = |a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    };
    // Owning heads8 exercises D512 MMA; heads4 covers its borrowed-KV consumer.
    for (layer, hd, heads, ring, window) in
        [(0, 256, 8, 1023, 512), (1, 512, 8, 0, 0), (1, 512, 4, 0, 0)]
    {
        let width = heads * hd;
        let kv_width = KV_HEADS * hd;
        let stride = u64::from(kv_width / 32 * 18);
        let cache_rows = if ring != 0 { ring + 1 } else { CAPACITY };
        let queries: Vec<Vec<f32>> = (0..2_u32)
            .map(|slot| {
                (0..width)
                    .map(|i| ((i * 13 + slot * 29) % 101) as f32 / 101.0 - 0.5)
                    .collect()
            })
            .collect();
        let caches: Vec<[Vec<u8>; 2]> = (0..2_u32)
            .map(|slot| {
                [
                    raw_q4(cache_rows, kv_width, 3 + slot * 11),
                    raw_q4(cache_rows, kv_width, 7 + slot * 13),
                ]
            })
            .collect();
        let upload = |slot: usize, offset: u64| {
            for is_v in [false, true] {
                be.write_kv_bytes(
                    layer,
                    is_v,
                    offset,
                    &caches[slot][usize::from(is_v)],
                );
            }
            be.end().unwrap();
        };
        let check_cache = |slot: usize, offset: u64| {
            for is_v in [false, true] {
                let expected = &caches[slot][usize::from(is_v)];
                let mut actual = vec![0; expected.len()];
                be.read_kv_bytes(layer, is_v, offset, &mut actual);
                be.end().unwrap();
                assert_eq!(actual.len(), expected.len());
                let first = actual.iter().zip(expected).position(|(a, b)| a != b);
                assert!(
                    first.is_none(),
                    "D{hd} heads={heads} slot={slot} V={is_v} cache byte {first:?}"
                );
            }
        };
        let ordinary = |slot: usize, decode: bool| {
            crate::correctness::write_f32_checked(BufId::Q, 0, &queries[slot]).unwrap();
            be.begin_forward(decode);
            // The final prompt row and first decode row straddle the 6144 seam.
            let pos = if decode { POS } else { PREFILL_POS };
            be.attention(
                layer,
                hd,
                heads,
                KV_HEADS,
                kv_width,
                pos,
                1.0,
                window,
                1,
                pos + 1,
                ring,
            );
            be.end().unwrap();
            read_output(width as usize)
        };
        let mut oracle = Vec::new();
        // Independent identity-layout oracle for each conversation's exact
        // Q/K/V bytes, before shared storage is installed in its final layout.
        for slot in 0..2_usize {
            assert!(be.select_slot(slot as u32));
            if ring == 0 {
                set_pages(&identity);
            }
            upload(slot, 0);
            check_cache(slot, 0);
            oracle.push([ordinary(slot, true), ordinary(slot, false)]);
        }
        let mut standalone = Vec::new();
        for slot in 0..2_usize {
            assert!(be.select_slot(slot as u32));
            let offset = if ring == 0 {
                slot as u64 * u64::from(CAPACITY) * stride
            } else {
                0
            };
            if ring == 0 {
                set_pages(if slot == 0 { &identity } else { &mapped });
            }
            upload(slot, offset);
            check_cache(slot, offset);
            standalone.push([ordinary(slot, true), ordinary(slot, false)]);
        }
        // The mapped slot1 is foreign during CoBatch, while slot0 is selected.
        assert!(be.select_slot(0));
        let q: Vec<f32> = queries.iter().flatten().copied().collect();
        crate::correctness::write_f32_checked(BufId::Q, 0, &q).unwrap();
        assert!(be.set_decode_rows(Some(imparo_backend::RowRoute::Fast)));
        be.begin_forward(false);
        let rows = [
            imparo_backend::SlotRow { slot: 0, pos: POS },
            imparo_backend::SlotRow { slot: 1, pos: POS },
        ];
        assert!(be.attention_slot_rows(
            layer,
            hd,
            heads,
            KV_HEADS,
            kv_width,
            1.0,
            window,
            &rows,
            &[POS + 1, POS + 1],
            ring
        ));
        be.end().unwrap();
        assert!(be.set_decode_rows(None));
        let actual = read_output((2 * width) as usize);
        let mut decode_route_diff = 0.0_f32;
        let mut prefill_route_diff = 0.0_f32;
        let mut row_diff = 0.0_f32;
        for slot in 0..2_usize {
            decode_route_diff = decode_route_diff
                .max(difference(&standalone[slot][0], &oracle[slot][0]));
            prefill_route_diff = prefill_route_diff
                .max(difference(&standalone[slot][1], &oracle[slot][1]));
            row_diff = row_diff.max(difference(
                &actual[slot * width as usize..(slot + 1) * width as usize],
                &standalone[slot][0],
            ));
        }
        eprintln!(
            "E4B retained Q4 D{hd} heads={heads} pos={POS} prefill_pos={PREFILL_POS} ring={ring}: identity/mapped decode maxdiff={decode_route_diff}, prefill-tail maxdiff={prefill_route_diff}, ordinary/CoBatch maxdiff={row_diff}"
        );
        // Verify the selected binding first, then the borrowed binding, and
        // ensure attention never mutates either conversation's raw cache.
        for slot in [0_usize, 1] {
            assert!(be.select_slot(slot as u32));
            let offset = if ring == 0 {
                slot as u64 * u64::from(CAPACITY) * stride
            } else {
                0
            };
            check_cache(slot, offset);
        }
        assert!(
            decode_route_diff <= 1e-6,
            "D{hd} heads={heads}: identical logical Q/K/V changed with decode page layout, maxdiff={decode_route_diff}"
        );
        assert!(
            prefill_route_diff <= 1e-6,
            "D{hd} heads={heads}: identical logical Q/K/V changed with prefill-tail page layout, maxdiff={prefill_route_diff}"
        );
        assert!(
            row_diff <= 1e-6,
            "D{hd} heads={heads}: CoBatch changed a standalone row, maxdiff={row_diff}"
        );
    }
}

#[test]
#[ignore = "requires an idle CUDA GPU and owns process-global native state"]
fn gpu_cobatch_recurrent_rows() {
    use imparo_backend::{ConvForm, DeltaNet, SlotStateRow};
    let mut weights = vec![0.125_f32; 8192];
    weights[4096] = -0.3;
    weights[4097] = -0.7;
    weights[4100] = 0.1;
    weights[4101] = -0.2;
    let weights = Box::leak(weights.into_boxed_slice());
    unsafe {
        CudaContext::get().init_weights(
            weights.as_ptr().cast(),
            (weights.len() * 4) as u64,
            &[],
        )
    }
    .unwrap();
    let be = CudaBackend;
    let state_len = 65568_usize;
    for (id, bytes) in [
        (BufId::Q, 4096_u64),
        (BufId::K, 64),
        (BufId::V, 64),
        (BufId::Attn, 2048),
        (BufId::Recur, (state_len * 4) as u64),
    ] {
        be.alloc(id, bytes).unwrap();
    }
    be.set_kv_types(1, 1);
    be.alloc_kv_layout(
        &[8192],
        &[KvLayout {
            layer: 0,
            logical_slots: 128,
            k_stride: 64,
            v_stride: 64,
            ..KvLayout::default()
        }],
    )
    .unwrap();
    be.end().unwrap();
    assert!(be.set_slots(2, &[]));
    let write =
        |id, data: &[f32]| crate::correctness::write_f32_checked(id, 0, data).unwrap();
    let read = |id, n: usize| {
        let mut v = vec![0.0; n];
        crate::correctness::read_f32_checked(id, 0, &mut v).unwrap();
        v
    };
    let equal = |a: &[f32], b: &[f32], what: &str| {
        assert_eq!(a.len(), b.len());
        let bad = a
            .iter()
            .zip(b)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(bad.is_none(), "{what}: first differing element {bad:?}");
    };
    let input = |n: usize, seed: usize| {
        (0..n)
            .map(|i| ((i * 13 + seed * 19) % 101) as f32 / 1000.0 - 0.05)
            .collect::<Vec<_>>()
    };
    for mode in 0..3 {
        let (iw, ow) = match mode {
            0 => (192, 64),
            1 => (64, 64),
            _ => (512, 256),
        };
        let q = input(2 * iw, 1);
        let alpha = input(4, 2);
        let beta = input(4, 3);
        let initial = [input(state_len, 7), input(state_len, 11)];
        let rows = [
            SlotStateRow {
                slot: 0,
                state_off: 0,
                state_out_off: 32780,
            },
            SlotStateRow {
                slot: 1,
                state_off: 5,
                state_out_off: 32785,
            },
        ];
        let form = if mode == 0 {
            ConvForm::GatedBcx
        } else {
            ConvForm::PlainSilu
        };
        let delta = |r: SlotStateRow| DeltaNet {
            qkv: BufId::Q,
            alpha: BufId::K,
            beta: BufId::V,
            a_off: 4096 * 4,
            dt_bias_off: 4100 * 4,
            state: BufId::Recur,
            state_off: r.state_off,
            state_out_off: r.state_out_off,
            out: BufId::Attn,
            epilogue: None,
            snap: None,
            k_heads: 1,
            v_heads: 2,
            key_dim: 128,
            value_dim: 128,
            n_tok: 1,
            eps: 1e-6,
        };
        let batch = |rs: &[SlotStateRow]| {
            if mode == 2 {
                be.delta_net_slot_rows(&delta(rows[0]), rs)
            } else {
                be.causal_conv_slot_rows(
                    form,
                    BufId::Q,
                    0,
                    BufId::Recur,
                    rs,
                    BufId::Attn,
                    64,
                    4,
                )
            }
        };
        let mut expected = Vec::new();
        let mut states = Vec::new();
        for slot in 0..2 {
            assert!(be.select_slot(slot as u32));
            write(BufId::Recur, &initial[slot]);
            write(BufId::Q, &q[slot * iw..(slot + 1) * iw]);
            write(BufId::K, &alpha[slot * 2..slot * 2 + 2]);
            write(BufId::V, &beta[slot * 2..slot * 2 + 2]);
            be.begin_forward(true);
            if mode == 2 {
                assert!(be.delta_net(&delta(rows[slot])));
            } else {
                be.causal_conv(
                    form,
                    BufId::Q,
                    0,
                    BufId::Recur,
                    rows[slot].state_off,
                    rows[slot].state_out_off,
                    BufId::Attn,
                    64,
                    4,
                    1,
                );
            }
            be.end().unwrap();
            expected.extend(read(BufId::Attn, ow));
            states.push(read(BufId::Recur, state_len));
            write(BufId::Recur, &initial[slot]);
            be.end().unwrap();
        }
        write(BufId::Q, &q);
        write(BufId::K, &alpha);
        write(BufId::V, &beta);
        be.end().unwrap();
        assert!(be.set_decode_rows(Some(imparo_backend::RowRoute::Fast)));
        let invalid = [
            rows[0],
            SlotStateRow {
                slot: 99,
                ..rows[1]
            },
        ];
        assert!(!batch(&invalid));
        assert!(!batch(&[rows[0], rows[0]]));
        assert!(be.set_decode_rows(None));
        for slot in 0..2 {
            assert!(be.select_slot(slot as u32));
            equal(
                &read(BufId::Recur, state_len),
                &initial[slot],
                "invalid batch changed history",
            );
        }
        assert!(be.set_decode_rows(Some(imparo_backend::RowRoute::Fast)));
        be.begin_forward(false);
        assert!(batch(&rows));
        be.end().unwrap();
        assert!(be.set_decode_rows(None));
        equal(&read(BufId::Attn, 2 * ow), &expected, "row output");
        // Slot1 is selected. Both input planes and untouched padding must remain intact.
        for slot in [1, 0] {
            assert!(be.select_slot(slot as u32));
            equal(
                &read(BufId::Recur, state_len),
                &states[slot],
                "slot history/commit plane",
            );
        }
        println!(
            "CoBatch recurrent mode={mode}: output and complete state match independent M1; invalid/duplicate slot preflight preserved state"
        );
    }
}
