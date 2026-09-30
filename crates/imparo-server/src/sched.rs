//! The engine loop (docs/continuous-batching.md, section 3): one thread owns the engine and
//! serves every chat request.
//!
//! ```text
//! iteration:  take new jobs -> resume parked rows that fit -> admit one (prefill; alone, it
//!             decodes on its own route) -> one decode step for every row of the co-batch
//!             -> hand out the tokens -> finish the rows whose replies ended
//! ```
//!
//! A request runs alone on today's route -- the mega kernel, pipelined decode, the drafter --
//! until another request is running or waiting. Then, at the next point a plain decode step
//! can carry it on from, it hands over and decodes as a row of the co-batch: one
//! `decode_rows` step per iteration for every running request.
//!
//! A row the pool has no page for is parked at its page boundary and resumes there, before
//! any new request starts (docs/continuous-batching.md, section 5).
//!
//! A request admitted while rows run is prefilled a chunk per iteration, after their decode
//! step, and joins them after its first token; no other request starts meanwhile.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use super::{
    Engine, Generation, Job, Outcome, PromptPrefill, Reply, Turn, abandon,
    chat_completions, finish, first_row, note_request_end, prefill_ended,
};

/// Requests admitted and not yet finished: while any runs, the idle release waits.
pub(super) static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// Requests that ended and rows that parked, since the start. A job that did not fit is
/// tried again once this has moved: only then can the pool have pages it did not have.
static LEFT: AtomicUsize = AtomicUsize::new(0);

/// What a request running alone needs to know about the rest of the loop.
pub(super) struct Sched<'a> {
    /// Slots the co-batch has; 1 runs requests one at a time.
    pub(super) slots: usize,
    /// Rows decoding in the co-batch.
    pub(super) running: usize,
    /// Jobs the loop holds that can start.
    pub(super) waiting: usize,
    /// Jobs submitted and not yet taken by the loop.
    pub(super) submitted: &'a AtomicUsize,
}

impl Sched<'_> {
    /// Another request is running or can start: a request running alone hands over to the
    /// co-batch at its next point a plain decode step can carry it on from.
    pub(super) fn others(&self) -> bool {
        self.slots > 1
            && (self.running > 0
                || self.waiting > 0
                || self.submitted.load(Ordering::Relaxed) > 0)
    }
}

/// A label of its own for a keyless request whose content-derived label a running request
/// holds.
pub(super) fn fresh_label(label: &str) -> String {
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    format!("{label}~{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

/// The slot a request for `label` decodes in: the one its conversation occupies, else the
/// selected one when nothing runs there (so a lone client's next turn finds its state where
/// it left it), else an empty one, else one whose occupant is idle. None when every slot is
/// running a request.
pub(super) fn choose_slot(
    pl: &imparo_kv::pool::PoolMode,
    label: &str,
    slots: usize,
) -> Option<usize> {
    if let Some(s) = pl.slot_of(label) {
        return Some(s);
    }
    let free = |s: usize| pl.occupant(s).is_none_or(|o| !pl.is_running(o));
    let selected = pl.selected_slot();
    if free(selected) {
        return Some(selected);
    }
    (0..slots)
        .find(|&s| pl.occupant(s).is_none())
        .or_else(|| (0..slots).find(|&s| free(s)))
}

/// A request admitted beside running rows, its prompt being prefilled a chunk per loop
/// iteration, between their decode steps (docs/continuous-batching.md, section 3). It joins
/// them after its first token. From its admission its conversation is pinned and its slot
/// taken.
pub(super) struct Prefilling {
    pub(super) turn: Turn,
    pub(super) prompt: PromptPrefill,
    /// Its slot: its conversation's tables and recurrent state.
    pub(super) slot: usize,
    /// The last chunk's logits: its first token is picked from them.
    pub(super) logits: Vec<f32>,
    pub(super) t_prefill: Instant,
    pub(super) reply: Reply,
    pub(super) temperature: f64,
}

/// A request decoding in the co-batch; the rest of its reply is made here, a token a step.
pub(super) struct Row {
    turn: Turn,
    generation: Generation,
    /// Its slot: its conversation's tables and recurrent state.
    slot: usize,
    /// The token the next step forwards.
    next: u32,
    /// The reply does not have `next` yet: the pipelined route learns a token one step late.
    untaken: bool,
    /// Positions filled: where the next step writes.
    pos: usize,
    /// Positions the conversation's blocks cover.
    room: usize,
    /// Steps it took with other rows (`decode_rows`), and alone on the one-row route.
    cobatched: usize,
    lone: usize,
}

impl Row {
    /// A request that stopped decoding alone, in the selected slot, becomes a row. Its
    /// conversation is pinned: from here the loop lets other requests run between its steps,
    /// and eviction, a switch and the idle release must leave it alone.
    pub(super) fn join(
        engine: &mut Engine,
        turn: Turn,
        generation: Generation,
        next: u32,
        taken: bool,
    ) -> Self {
        let pos = engine.model.kv_runtime().filled;
        let (slot, room) = match (engine.pool.as_mut(), turn.label.as_deref()) {
            (Some(pl), Some(label)) => {
                pl.pin(label);
                (pl.selected_slot(), pl.room(label))
            }
            _ => (0, 0),
        };
        if imparo_model::log_on() {
            eprintln!(
                "[imparo] cobatch join: conv={} slot={slot} pos={pos}",
                turn.label.as_deref().unwrap_or("-")
            );
        }
        Self {
            turn,
            generation,
            slot,
            next,
            untaken: !taken,
            pos,
            room,
            cobatched: 0,
            lone: 0,
        }
    }

    /// The tokens with a KV row: the prompt, then the reply's tokens but the last, which is
    /// `next` and has none yet.
    fn stream(&self) -> std::io::Result<Vec<u32>> {
        let generated = &self.generation.reply.generated;
        if self.untaken
            || self.turn.ids.len() + generated.len() != self.pos + 1
            || generated.last() != Some(&self.next)
        {
            return Err(std::io::Error::other(format!(
                "row at {} has {} prompt and {} reply tokens",
                self.pos,
                self.turn.ids.len(),
                generated.len()
            )));
        }
        let mut stream = self.turn.ids.clone();
        stream.extend_from_slice(&generated[..generated.len() - 1]);
        Ok(stream)
    }
}

/// The engine loop: the only caller of `chat_completions`. Returns when every sender of
/// `jobs` has gone.
pub(super) fn run(
    engine: &Mutex<Engine>,
    jobs: &Receiver<Job>,
    submitted: &AtomicUsize,
) {
    let mut waiting: VecDeque<Job> = VecDeque::new();
    // Jobs for a conversation a running request holds: a client's two turns of one
    // conversation cannot run at once.
    let mut blocked: Vec<Job> = Vec::new();
    let mut rows: Vec<Row> = Vec::new();
    // Rows the pool had no page for, oldest first. They resume before any job starts.
    let mut parked: VecDeque<Row> = VecDeque::new();
    // `LEFT` when the job at the head of the line did not fit: it waits until that moves.
    let mut stalled: Option<usize> = None;
    // A request admitted beside running rows, prefilled a chunk per iteration. No other
    // request starts until it has joined them.
    let mut prefilling: Option<Box<Prefilling>> = None;
    loop {
        // What this iteration's engine calls autorelease is freed at its end: the loop's
        // thread never exits, so nothing else would drain it.
        let _pool = imparo_model::host::AutoreleaseScope::open();
        if rows.is_empty()
            && parked.is_empty()
            && prefilling.is_none()
            && waiting.is_empty()
            && blocked.is_empty()
        {
            // Nothing to do until a job arrives.
            let Ok(job) = jobs.recv() else {
                return;
            };
            submitted.fetch_sub(1, Ordering::Relaxed);
            waiting.push_back(job);
        }
        while let Ok(job) = jobs.try_recv() {
            submitted.fetch_sub(1, Ordering::Relaxed);
            waiting.push_back(job);
        }
        let mut guard = engine.lock().unwrap_or_else(PoisonError::into_inner);
        let e = &mut *guard;
        // A blocked job whose conversation no longer runs goes back to the head of the line.
        // Only a row, running or parked, or a request being prefilled holds a conversation, so
        // with none every blocked job is ready.
        let idle = rows.is_empty() && parked.is_empty() && prefilling.is_none();
        let (ready, held): (Vec<Job>, Vec<Job>) =
            blocked.drain(..).partition(|job| idle || !busy(e, job));
        blocked = held;
        for job in ready.into_iter().rev() {
            waiting.push_front(job);
        }
        // Parked rows first, oldest first. Beside running requests one needs its pages and
        // one more per running request, its own included, as a new request does. A request
        // being prefilled runs and holds a slot.
        loop {
            let running = rows.len() + usize::from(prefilling.is_some());
            let Some(row) = parked.front() else {
                break;
            };
            if running >= e.slots || (running > 0 && !fits(e, row.pos + 1, running)) {
                break;
            }
            let Some(mut row) = parked.pop_front() else {
                break;
            };
            match resume(e, &mut row) {
                Ok(()) => rows.push(row),
                Err(Resume::NoSlot) => {
                    parked.push_front(row);
                    break;
                }
                Err(Resume::Failed(err)) => drop_row(e, row, err),
            }
        }
        if stalled.is_some_and(|at| at != LEFT.load(Ordering::Relaxed)) {
            stalled = None;
        }
        if parked.is_empty()
            && stalled.is_none()
            && prefilling.is_none()
            && rows.len() < e.slots
        {
            while let Some(job) = waiting.pop_front() {
                if !rows.is_empty() && busy(e, &job) {
                    blocked.push(job);
                    continue;
                }
                let sched = Sched {
                    slots: e.slots,
                    running: rows.len(),
                    waiting: waiting.len(),
                    submitted,
                };
                if let Some(job) = admit(e, job, &sched, &mut rows, &mut prefilling) {
                    waiting.push_front(job);
                    stalled = Some(LEFT.load(Ordering::Relaxed));
                }
                break;
            }
        }
        if !rows.is_empty() {
            let alone = waiting.is_empty()
                && parked.is_empty()
                && submitted.load(Ordering::Relaxed) == 0;
            step(e, &mut rows, &mut parked, alone, prefilling.is_some());
        }
        // After the rows' step, one chunk of the request being prefilled; after its last, it
        // takes its first token and joins them.
        if let Some(p) = prefilling.as_deref_mut() {
            match prefill_chunk(e, p) {
                Ok(false) => {}
                Ok(true) => {
                    if let Some(p) = prefilling.take() {
                        join_prefilled(e, *p, &mut rows);
                    }
                }
                Err(err) => {
                    if let Some(p) = prefilling.take() {
                        let streaming = p.reply.streaming;
                        let Prefilling { mut turn, slot, .. } = *p;
                        fail(e, slot, &mut turn, streaming, err);
                    }
                }
            }
        }
        drop(guard);
        // The loop takes the lock back at once; a thread waiting for it (an erase, a
        // barrier, the idle release, shutdown) gets it first.
        while ENGINE_WANTED.load(Ordering::Acquire) > 0 {
            std::thread::yield_now();
        }
    }
}

/// Threads other than the loop waiting for the engine.
static ENGINE_WANTED: AtomicUsize = AtomicUsize::new(0);

/// The engine, for a thread other than the loop: the loop lets it in between two of its
/// iterations.
pub(super) fn lock_engine(engine: &Mutex<Engine>) -> std::sync::MutexGuard<'_, Engine> {
    ENGINE_WANTED.fetch_add(1, Ordering::AcqRel);
    let guard = engine.lock().unwrap_or_else(PoisonError::into_inner);
    ENGINE_WANTED.fetch_sub(1, Ordering::AcqRel);
    guard
}

/// A keyed job whose conversation a running or parked request holds.
fn busy(e: &Engine, job: &Job) -> bool {
    job.req
        .header("x-conversation-id")
        .filter(|c| *c != "default")
        .is_some_and(|c| e.pool.as_ref().is_some_and(|pl| pl.is_running(c)))
}

/// Whether the pool can obtain the pages a row reaching `upper` positions takes, plus one
/// per running row (docs/continuous-batching.md, section 4).
fn fits(e: &Engine, upper: usize, running: usize) -> bool {
    e.pool.as_ref().is_some_and(|pl| {
        pl.obtainable_units() >= upper.div_ceil(imparo_kv::grid_tokens()) + running
    })
}

/// Runs a job: alone to its end, or until it hands over and joins `rows`; beside running rows,
/// up to its prefill, which goes on in `prefilling`. Returns the job when the pool cannot hold
/// it beside the rows running; nothing of it ran.
fn admit(
    e: &mut Engine,
    job: Job,
    sched: &Sched<'_>,
    rows: &mut Vec<Row>,
    prefilling: &mut Option<Box<Prefilling>>,
) -> Option<Job> {
    IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
    match chat_completions(job, e, sched) {
        Ok(Outcome::Done) => ended(),
        Ok(Outcome::Wait(job)) => {
            IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
            return Some(job);
        }
        Ok(Outcome::Row(row)) => seat(e, *row, rows),
        Ok(Outcome::Prefill(p)) => {
            if imparo_model::log_on() {
                eprintln!(
                    "[imparo] cobatch prefill: conv={} slot={} from={} prompt={} rows={}",
                    p.turn.label.as_deref().unwrap_or("-"),
                    p.slot,
                    p.turn.start_pos,
                    p.turn.prompt_tokens,
                    rows.len()
                );
            }
            *prefilling = Some(p);
        }
        Err(err) => {
            eprintln!("[imparo] connection error: {err}");
            ended();
        }
    }
    None
}

/// A row whose reply does not have its pending token yet takes it: the row runs on, or ends
/// when that token ends its reply.
fn seat(e: &mut Engine, mut row: Row, rows: &mut Vec<Row>) {
    if !row.untaken {
        rows.push(row);
        return;
    }
    row.untaken = false;
    match row
        .generation
        .reply
        .take(row.next, &e.tok, &mut row.turn.out)
    {
        Ok(false) => rows.push(row),
        Ok(true) => {
            let others = !rows.is_empty();
            end_row(e, row, others);
        }
        Err(err) => drop_row(e, row, err),
    }
}

/// One chunk of the prompt being prefilled, in its slot. True once its last has run.
fn prefill_chunk(e: &mut Engine, p: &mut Prefilling) -> std::io::Result<bool> {
    select(e, p.slot)?;
    let Engine { model, pool, .. } = e;
    p.prompt.step(
        &mut **model,
        pool.as_mut(),
        &p.turn.ids,
        &mut p.logits,
        None,
    )
}

/// A request whose last chunk has run: it takes its first token and joins the rows, or ends
/// when that token ends its reply. A reply of no tokens (a seed) ends before taking one.
fn join_prefilled(e: &mut Engine, p: Prefilling, rows: &mut Vec<Row>) {
    match prefill_ended(e, &p) {
        Ok((prefill_ms, witness)) => {
            let empty = p.reply.max_tokens == 0;
            let row = first_row(e, p, prefill_ms, witness);
            if empty {
                let others = !rows.is_empty();
                end_row(e, row, others);
            } else {
                seat(e, row, rows);
            }
        }
        Err(err) => {
            let streaming = p.reply.streaming;
            let Prefilling { mut turn, slot, .. } = p;
            fail(e, slot, &mut turn, streaming, err);
        }
    }
}

fn ended() {
    note_request_end();
    IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
    LEFT.fetch_add(1, Ordering::Relaxed);
}

/// Selects `slot` on the model and the pool together: every per-conversation read and write
/// from here is that slot's.
fn select(e: &mut Engine, slot: usize) -> std::io::Result<()> {
    e.model
        .select_slot(u32::try_from(slot).map_err(std::io::Error::other)?)
        .map_err(std::io::Error::other)?;
    if let Some(pl) = e.pool.as_mut() {
        pl.select_slot(slot);
    }
    Ok(())
}

/// One decode step for every row, then each row's token handed to its reply. `alone`:
/// nothing waits to start, so a single row takes the one-row route. `prefilling`: a request
/// being prefilled runs beside the rows.
fn step(
    e: &mut Engine,
    rows: &mut Vec<Row>,
    parked: &mut VecDeque<Row>,
    alone: bool,
    prefilling: bool,
) {
    // Room for the position each row writes. A row the pool has no page for parks while
    // others run; one that cannot have it otherwise fails by itself.
    let mut i = 0;
    while i < rows.len() {
        if rows[i].pos < rows[i].room {
            i += 1;
            continue;
        }
        let others = rows.len() > 1 || prefilling;
        match grow(e, &mut rows[i], others) {
            Ok(()) => i += 1,
            Err(Grow::Park) => {
                let row = rows.remove(i);
                match park(e, &row) {
                    Ok(()) => parked.push_back(row),
                    Err(err) => drop_row(e, row, err),
                }
            }
            Err(Grow::Fail(err)) => {
                let row = rows.remove(i);
                drop_row(e, row, err);
            }
        }
    }
    if rows.is_empty() {
        return;
    }
    let lone = alone && rows.len() == 1;
    let picks = if lone {
        // A lone row goes back to the one-row route, the mega kernel where there is one.
        let (slot, next, pos) = (rows[0].slot, rows[0].next, rows[0].pos);
        select(e, slot).and_then(|()| {
            e.model
                .forward_next(next, pos)
                .map(|pick| vec![pick])
                .map_err(std::io::Error::other)
        })
    } else {
        let batch: Vec<(u32, u32)> = rows
            .iter()
            .map(|r| (u32::try_from(r.slot).unwrap_or(u32::MAX), r.next))
            .collect();
        let probe = step_timing::begin();
        let picks = e
            .model
            .decode_rows(&batch, None, imparo_backend::RowRoute::Fast)
            .map_err(std::io::Error::other);
        if let Some(t0) = probe {
            step_timing::end(t0, batch.len());
        }
        picks
    };
    let picks = match picks {
        Ok(picks) if picks.len() == rows.len() => picks,
        Ok(picks) => {
            let err = std::io::Error::other(format!(
                "{} picks for {} rows",
                picks.len(),
                rows.len()
            ));
            fail_all(e, rows, &err);
            return;
        }
        Err(err) => {
            fail_all(e, rows, &err);
            return;
        }
    };
    for (row, pick) in rows.iter_mut().zip(picks) {
        row.next = pick;
        row.pos += 1;
        row.generation.decode_steps += 1;
        if lone {
            row.lone += 1;
        } else {
            row.cobatched += 1;
        }
    }
    let mut i = 0;
    while i < rows.len() {
        let row = &mut rows[i];
        match row
            .generation
            .reply
            .take(row.next, &e.tok, &mut row.turn.out)
        {
            Ok(false) => i += 1,
            Ok(true) => {
                let row = rows.remove(i);
                let others = !rows.is_empty() || prefilling;
                end_row(e, row, others);
            }
            Err(err) => {
                let row = rows.remove(i);
                drop_row(e, row, err);
            }
        }
    }
}

/// IMPARO_COBATCH_TIMING=1: per co-batched step, the wall time of `decode_rows` (encode, GPU,
/// wait, readback), its GPU time, and the host time since the previous step returned -- the
/// time the GPU sits idle between two steps. Every 64 steps a line of medians on stderr. Off, the
/// whole probe is one flag test.
mod step_timing {
    use std::sync::Mutex;
    use std::time::Instant;

    struct State {
        last_end: Option<Instant>,
        rows: Vec<usize>,
        wall_ms: Vec<f64>,
        gpu_ms: Vec<f64>,
        gap_ms: Vec<f64>,
    }
    static STATE: Mutex<State> = Mutex::new(State {
        last_end: None,
        rows: Vec::new(),
        wall_ms: Vec::new(),
        gpu_ms: Vec::new(),
        gap_ms: Vec::new(),
    });

    fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("IMPARO_COBATCH_TIMING").is_ok_and(|v| v == "1")
        })
    }

    pub(super) fn begin() -> Option<Instant> {
        on().then(Instant::now)
    }

    pub(super) fn end(t0: Instant, rows: usize) {
        let now = Instant::now();
        let gpu =
            imparo_model::backend::active().map_or(0.0, |b| b.last_gpu_us() / 1e3);
        let mut st = STATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let gap = st
            .last_end
            .map_or(f64::NAN, |e| (t0 - e).as_secs_f64() * 1e3);
        st.last_end = Some(now);
        st.rows.push(rows);
        st.wall_ms.push((now - t0).as_secs_f64() * 1e3);
        st.gpu_ms.push(gpu);
        if gap.is_finite() {
            st.gap_ms.push(gap);
        }
        if st.wall_ms.len() == 64 {
            let med = |v: &mut Vec<f64>| {
                v.sort_by(f64::total_cmp);
                v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
            };
            let rows_med = {
                st.rows.sort_unstable();
                st.rows[st.rows.len() / 2]
            };
            let (w, g) = {
                let State {
                    wall_ms, gpu_ms, ..
                } = &mut *st;
                (med(wall_ms), med(gpu_ms))
            };
            let gap = med(&mut st.gap_ms);
            eprintln!(
                "[imparo] cobatch timing: 64 steps, rows p50 {rows_med}, step {w:.2} ms, gpu \
{g:.2} ms, host gap before a step {gap:.2} ms"
            );
            st.rows.clear();
            st.wall_ms.clear();
            st.gpu_ms.clear();
            st.gap_ms.clear();
        }
    }
}

/// A step that failed as a whole fails every row in it.
fn fail_all(e: &mut Engine, rows: &mut Vec<Row>, err: &std::io::Error) {
    eprintln!(
        "[imparo] co-batched step failed ({err}); {} requests abandoned",
        rows.len()
    );
    for row in rows.drain(..) {
        drop_row(e, row, std::io::Error::other(err.to_string()));
    }
}

/// Why a row did not get the page it reached.
enum Grow {
    /// The pool has none to give while other rows run: the row parks.
    Park,
    Fail(std::io::Error),
}

/// Room for the position a row writes next: the pool hands out its next page and installs
/// the longer table in the row's slot.
fn grow(e: &mut Engine, row: &mut Row, others: bool) -> Result<(), Grow> {
    select(e, row.slot).map_err(Grow::Fail)?;
    let Engine {
        model,
        pool,
        store,
        disk,
        ..
    } = e;
    let (Some(pl), Some(label)) = (pool.as_mut(), row.turn.label.as_deref()) else {
        return Err(Grow::Fail(std::io::Error::other(
            "a co-batched row has no pool conversation",
        )));
    };
    // No free page, and every conversation the pool could drop runs or sits in a slot. With
    // others running, one of them ending gives pages back: park here, where the row's state
    // is a checkpoint the disk tier holds. A row running alone would wait for nothing.
    if others && store.is_some() && disk.is_some() && pl.obtainable_units() == 0 {
        return Err(Grow::Park);
    }
    let reach = (row.pos + 1).min(model.kv_runtime().capacity);
    row.room = pl
        .grow_room(&mut **model, label, reach, store.as_ref(), disk.as_ref())
        .map_err(|err| Grow::Fail(std::io::Error::other(err)))?;
    Ok(())
}

/// Parks a row at its page boundary (docs/continuous-batching.md, section 5): its turn so far
/// is sealed and written through, and its conversation leaves the slot, unpinned, so its
/// pages can be dropped. The row keeps its reply and its client.
fn park(e: &mut Engine, row: &Row) -> std::io::Result<()> {
    select(e, row.slot)?;
    let stream = row.stream()?;
    let Engine {
        model,
        pool,
        store,
        disk,
        ..
    } = e;
    let filled = model.kv_runtime().filled;
    match (pool.as_mut(), row.turn.label.as_deref()) {
        _ if filled != row.pos => {
            Err(format!("its slot holds {filled}, the row {}", row.pos))
        }
        (Some(pl), Some(label)) => {
            pl.park(&**model, store.as_ref(), disk.as_ref(), label, stream)
        }
        _ => Err("a co-batched row has no pool conversation".into()),
    }
    .map_err(|err| std::io::Error::other(format!("park: {err}")))?;
    if imparo_model::log_on() {
        eprintln!(
            "[imparo] cobatch park: conv={} slot={} pos={}",
            row.turn.label.as_deref().unwrap_or("-"),
            row.slot,
            row.pos
        );
    }
    LEFT.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// Why a parked row did not resume.
enum Resume {
    /// Every slot runs a request; it stays parked.
    NoSlot,
    /// In the slot it was given, which is its own now.
    Failed(std::io::Error),
}

/// Brings a parked row back at the position it parked at: a slot, and its KV and recurrent
/// state as they were there, with nothing forwarded. Its next step forwards its pending
/// token, the step it would have taken.
fn resume(e: &mut Engine, row: &mut Row) -> Result<(), Resume> {
    let slot = match (e.pool.as_ref(), row.turn.label.as_deref()) {
        (Some(pl), Some(label)) => choose_slot(pl, label, e.slots),
        _ => None,
    };
    let Some(slot) = slot else {
        return Err(Resume::NoSlot);
    };
    // The slot's own state takes its pages before its buffers are made; without room for
    // them the row stays parked, as without a slot.
    if let (Some(pl), Some(label)) = (e.pool.as_mut(), row.turn.label.as_deref()) {
        match pl.charge_slot(slot, label) {
            Ok(true) => {}
            Ok(false) => return Err(Resume::NoSlot),
            Err(err) => return Err(Resume::Failed(std::io::Error::other(err))),
        }
    }
    row.slot = slot;
    select(e, slot).map_err(Resume::Failed)?;
    let stream = row.stream().map_err(Resume::Failed)?;
    let keyless = row.turn.conversation == "default";
    let Engine {
        model,
        pool,
        store,
        disk,
        ..
    } = e;
    let upper = (row.pos + 1).min(model.kv_runtime().capacity);
    let room = match (pool.as_mut(), row.turn.label.as_deref()) {
        (Some(pl), Some(label)) => pl
            .resume_parked(
                &mut **model,
                store.as_ref(),
                disk.as_ref(),
                label,
                &stream,
                upper,
                keyless,
            )
            .map(|()| {
                pl.pin(label);
                pl.room(label)
            }),
        _ => Err("a co-batched row has no pool conversation".into()),
    }
    .map_err(|err| Resume::Failed(std::io::Error::other(format!("resume: {err}"))))?;
    let filled = model.kv_runtime().filled;
    if filled != row.pos {
        return Err(Resume::Failed(std::io::Error::other(format!(
            "resume: its slot holds {filled}, the row {}",
            row.pos
        ))));
    }
    row.room = room;
    if imparo_model::log_on() {
        eprintln!(
            "[imparo] cobatch resume: conv={} slot={slot} pos={}",
            row.turn.label.as_deref().unwrap_or("-"),
            row.pos
        );
    }
    Ok(())
}

/// A row whose reply has ended: its response and its turn written through. With `others`
/// still running, its conversation leaves the slot; the last one running stays, as a lone
/// request's does, so its client's next turn finds it there.
fn end_row(e: &mut Engine, mut row: Row, others: bool) {
    if let Err(err) = select(e, row.slot) {
        drop_row(e, row, err);
        return;
    }
    row.generation.decode_ms = row.generation.t_decode.elapsed().as_secs_f64() * 1e3;
    let label = row.turn.label.clone();
    if imparo_model::log_on() {
        eprintln!(
            "[imparo] cobatch end: conv={} cobatched_steps={} lone_steps={} leaves={others}",
            label.as_deref().unwrap_or("-"),
            row.cobatched,
            row.lone
        );
    }
    if let Err(err) = finish(e, row.turn, row.generation, !others) {
        eprintln!("[imparo] connection error: {err}");
    }
    let Engine {
        model,
        pool,
        store,
        disk,
        ..
    } = e;
    if let (Some(pl), Some(label)) = (pool.as_mut(), label.as_deref()) {
        pl.unpin(label);
        if others {
            if let Err(err) = pl.leave_slot(&**model, store.as_ref(), disk.as_ref()) {
                eprintln!("[imparo] kv leave slot: {err}");
            }
        }
    }
    ended();
}

/// A row that failed: its conversation is forgotten, and its client told when it can be.
fn drop_row(e: &mut Engine, mut row: Row, err: std::io::Error) {
    let streaming = row.generation.reply.streaming;
    fail(e, row.slot, &mut row.turn, streaming, err);
}

/// A request in `slot` that failed, decoding or being prefilled: its conversation is
/// forgotten, and its client told when it can be.
fn fail(
    e: &mut Engine,
    slot: usize,
    turn: &mut Turn,
    streaming: bool,
    err: std::io::Error,
) {
    let selected = select(e, slot);
    let label = turn.label.clone();
    if let (Some(pl), Some(label)) = (e.pool.as_mut(), label.as_deref()) {
        pl.unpin(label);
    }
    let result = match selected {
        Ok(()) => abandon(e, label.as_deref(), false, streaming, &mut turn.out, err),
        Err(select_err) => {
            // The slot's own state cannot be reached; the conversation still must not stay.
            if let (Some(pl), Some(label)) = (e.pool.as_mut(), label.as_deref()) {
                let _ = pl.forget(&[label.to_owned()]);
            }
            Err(select_err)
        }
    };
    if let Err(err) = result {
        eprintln!("[imparo] connection error: {err}");
    }
    ended();
}
