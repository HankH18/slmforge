"""The stall detector has to be trustworthy when nobody is watching it.

Its whole job is to break a silence. So the assertions that matter are the ones
proving it SPEAKS: on a hung-but-alive process, on a dead one, on a probe that
itself broke. A watcher that stays quiet through a crash is worse than no
watcher, because quiet reads as calm.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "watch_job.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("watch_job", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["watch_job"] = m
    spec.loader.exec_module(m)
    return m


def _run_watcher(args, timeout=30):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=timeout,
                          cwd=str(REPO_ROOT))


def _kinds(stdout):
    return [line.split()[0] for line in stdout.splitlines() if line.strip()]


# ----------------------------------------------------------------- the probe
def test_the_probe_reads_a_number(mod):
    assert mod.probe_value("echo 42")[0] == 42
    assert mod.probe_value("printf '  7\\n'")[0] == 7


def test_a_probe_that_prints_no_number_is_reported_not_guessed(mod):
    value, err = mod.probe_value("echo hello")
    assert value is None and "no number" in err


def test_a_probe_that_fails_is_reported(mod):
    value, err = mod.probe_value("exit 3")
    assert value is None and "exited 3" in err


def test_a_broken_probe_never_looks_like_a_calm_job(tmp_path):
    """A probe that errors and a job that is idle produce the same NUMBER --
    none. They must not produce the same EVENT, or the operator debugs the
    wrong thing."""
    r = _run_watcher(["--label", "t", "--probe", "exit 9",
                      "--poll-seconds", "0.2", "--stall-seconds", "1",
                      "--max-seconds", "1.5"])
    assert "PROBE-FAIL" in r.stdout, r.stdout


# ------------------------------------------------------------------ liveness
def test_pid_alive_tracks_reality(mod):
    import os
    assert mod.pid_alive(os.getpid()) is True
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    # a reaped child is gone; give the OS a beat on slower machines
    for _ in range(20):
        if not mod.pid_alive(proc.pid):
            break
        time.sleep(0.05)
    assert mod.pid_alive(proc.pid) is False


# ------------------------------------------------------------------ the point
def test_a_hung_but_alive_process_raises_a_stall(tmp_path):
    """THE ONE THIS EXISTS FOR. A `modal run` once hung after a remote failure
    and a watcher sat on it for ~69 minutes because the process was still
    alive. Liveness must not be mistaken for progress."""
    work = tmp_path / "w"
    work.mkdir()
    (work / "a").write_text("")
    hung = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        r = _run_watcher(["--label", "hung",
                          "--probe", f"ls {work} | wc -l",
                          "--pid", str(hung.pid),
                          "--poll-seconds", "0.2", "--stall-seconds", "1",
                          "--max-seconds", "3"])
        assert "STALL" in _kinds(r.stdout), r.stdout
        stall = [l for l in r.stdout.splitlines() if l.startswith("STALL")][0]
        assert "process=alive" in stall, (
            "a stall on a LIVE process must say so -- that distinction is the "
            "difference between 'wait longer' and 'kill it'")
    finally:
        hung.kill()
        hung.wait()


def test_a_finished_process_ends_the_watch(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    (work / "a").write_text("")
    quick = subprocess.Popen([sys.executable, "-c", "pass"])
    quick.wait()
    r = _run_watcher(["--label", "done", "--probe", f"ls {work} | wc -l",
                      "--pid", str(quick.pid),
                      "--poll-seconds", "0.2", "--stall-seconds", "30",
                      "--max-seconds", "3"])
    assert "EXIT" in _kinds(r.stdout), r.stdout
    assert r.returncode == 0


def test_the_exit_event_refuses_to_imply_success(tmp_path):
    """The watcher cannot see a non-child's exit STATUS. Saying only 'finished'
    would let a crashed job read as a completed one."""
    work = tmp_path / "w"
    work.mkdir()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    r = _run_watcher(["--label", "d", "--probe", f"ls {work} | wc -l",
                      "--pid", str(dead.pid), "--poll-seconds", "0.2",
                      "--max-seconds", "3"])
    exit_line = [l for l in r.stdout.splitlines() if l.startswith("EXIT")][0]
    assert "not observable" in exit_line and "log" in exit_line


def test_progress_resumes_the_watch_and_is_announced(tmp_path):
    """A stall that resolves must be retracted, or the operator kills a job
    that recovered."""
    work = tmp_path / "w"
    work.mkdir()
    script = (f"import time,pathlib\n"
              f"p=pathlib.Path({str(work)!r})\n"
              f"time.sleep(1.2)\n"
              f"(p/'late').write_text('')\n"
              f"time.sleep(5)\n")
    job = subprocess.Popen([sys.executable, "-c", script])
    try:
        r = _run_watcher(["--label", "rec", "--probe", f"ls {work} | wc -l",
                          "--pid", str(job.pid), "--poll-seconds", "0.2",
                          "--stall-seconds", "0.6", "--progress-every", "0.1",
                          "--max-seconds", "3"])
        kinds = _kinds(r.stdout)
        assert "STALL" in kinds and "RECOVER" in kinds, r.stdout
        assert kinds.index("STALL") < kinds.index("RECOVER")
    finally:
        job.kill()
        job.wait()


def test_reaching_the_target_ends_the_watch(tmp_path):
    """The target must be REACHED, i.e. the watcher must actually observe the
    work happen.

    JUSTIFIED EDIT (was: all 5 files created before the watch began). The old
    setup pre-satisfied --target on the very first probe read, so it asserted
    that a watcher which observed nothing may still exit 0. That is the false-
    success path -- a stale 147-line file from a previous run certified a dead
    job as complete -- and it is now pinned separately, and safely, by
    test_a_target_already_satisfied_at_start_is_refused. This test keeps the
    contract its own name states: work happens, TARGET fires, exit 0.
    """
    work = tmp_path / "w"
    work.mkdir()
    (work / "0").write_text("")
    filler = (f"import time,pathlib\n"
              f"p=pathlib.Path({str(work)!r})\n"
              f"time.sleep(0.6)\n"
              f"[ (p/str(i)).write_text('') for i in range(1,5) ]\n"
              f"time.sleep(10)\n")
    job = subprocess.Popen([sys.executable, "-c", filler])
    try:
        r = _run_watcher(["--label", "t", "--probe", f"ls {work} | wc -l",
                          "--target", "5", "--poll-seconds", "0.2",
                          "--max-seconds", "6"], timeout=40)
        assert "TARGET" in _kinds(r.stdout), r.stdout
        assert r.returncode == 0, r.stdout
    finally:
        job.kill()
        job.wait()


# --------------------------------------------------------------- the log scan
def test_failure_lines_in_the_log_are_surfaced(tmp_path, mod):
    """A job can fail loudly in its log while the probe is still frozen. The
    log is a SECOND source of truth, not a replacement for the probe."""
    log = tmp_path / "job.log"
    log.write_text("all fine\nTraceback (most recent call last):\nok again\n")
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.scan_log(log, 0, "x")
    assert "LOGFAIL" in buf.getvalue() and "Traceback" in buf.getvalue()


def test_the_log_scan_only_reports_new_lines(tmp_path, mod):
    log = tmp_path / "job.log"
    log.write_text("Error: one\n")
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        offset = mod.scan_log(log, 0, "x")
        mod.scan_log(log, offset, "x")     # nothing new
    assert buf.getvalue().count("LOGFAIL") == 1


def test_a_missing_log_does_not_kill_the_watcher(tmp_path, mod):
    assert mod.scan_log(tmp_path / "nope.log", 0, "x") == 0


def test_nonsense_intervals_are_refused():
    assert _run_watcher(["--label", "x", "--probe", "echo 1",
                         "--poll-seconds", "0"]).returncode == 2
    assert _run_watcher(["--label", "x", "--probe", "echo 1",
                         "--stall-seconds", "-5"]).returncode == 2


# =============================================================================
# REGRESSION TESTS FOR THE AUDIT FINDINGS
#
# This script is installed into a GLOBAL config and runs unattended in every
# project, against long expensive jobs. So the bar below is not "the happy path
# works" -- it is "this thing cannot damage a machine, and cannot lie by staying
# quiet". Each test names the failure it prevents.
# =============================================================================
import contextlib
import io
import os
import re
import signal
import textwrap
from collections import OrderedDict


@pytest.fixture
def fresh():
    """A private module instance, so a test that fakes the clock or the probe
    cannot leak that into another test."""
    spec = importlib.util.spec_from_file_location("watch_job_fresh", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _drive(mod, argv, probe_values, wall_jump_at=None, wall_jump=0.0,
           poll=30.0, max_polls=100000):
    """Run main() on a VIRTUAL monotonic clock with a scripted probe.

    Lets a 12-hour watch be asserted in milliseconds, which is the only way to
    test stall backoff and heartbeats honestly.
    """
    clock = [1_000_000.0]
    wall = [1_700_000_000.0]
    polls = [0]

    def fake_sleep(seconds):
        clock[0] += seconds
        wall[0] += seconds
        polls[0] += 1
        if wall_jump_at is not None and polls[0] == wall_jump_at:
            wall[0] += wall_jump          # laptop lid / NTP step: WALL only
        if polls[0] > max_polls:
            raise AssertionError("watch did not terminate")

    class _FakeTime:
        """Rebinds the MODULE'S `time` name only.

        Assigning to mod.time.sleep would mutate the one shared `time` module
        and silently break every later test in the process -- it did exactly
        that, and made two real signal tests fail for a fake reason.
        """
        sleep = staticmethod(fake_sleep)
        monotonic = staticmethod(lambda: clock[0])
        time = staticmethod(lambda: wall[0])

        def __getattr__(self, name):
            return getattr(time, name)

    mod.time = _FakeTime()

    seq = iter(probe_values)

    def fake_probe(cmd, timeout=60.0, parse="strict"):
        try:
            v = next(seq)
        except StopIteration:
            v = probe_values[-1] if probe_values else None
        return (None, v) if isinstance(v, str) else (v, None)

    mod.probe_value = fake_probe

    buf = io.StringIO()
    code = None
    with contextlib.redirect_stdout(buf):
        try:
            code = mod.run(list(argv) + ["--poll-seconds", str(poll)])
        except SystemExit as e:
            code = e.code
    return buf.getvalue(), code


def _counts(text):
    out = {}
    for line in text.splitlines():
        if line.strip():
            out[line.split()[0]] = out.get(line.split()[0], 0) + 1
    return out


# ------------------------------------------------- priority 1: system damage
def test_pid_alive_refuses_process_group_targets(mod):
    """os.kill(0, sig) hits the CALLER'S WHOLE PROCESS GROUP and os.kill(-1,
    sig) hits every process the user may signal. With signal 0 those are only
    existence probes -- which silently made the watch immortal -- but this
    function must never accept a group target, because the moment anyone puts a
    real signal here, --pid 0 becomes a process-group kill of the user's own
    session."""
    for bad in (0, -1, -99999):
        with pytest.raises(ValueError):
            mod.pid_alive(bad)


def test_the_cli_refuses_a_nonpositive_pid():
    for bad in ("0", "-1"):
        r = _run_watcher(["--label", "x", "--probe", "echo 1", "--pid", bad])
        assert r.returncode == 2, r.stdout
        assert "positive process id" in r.stderr


def test_a_pid_larger_than_the_kernel_allows_does_not_kill_the_watcher(mod):
    """os.kill raises OverflowError, not ValueError, for a pid past C's pid_t.
    It used to escape pid_alive's except clause and abort the watch mid-job."""
    assert mod.pid_alive(2 ** 63) is False
    assert mod.pid_alive(10 ** 30) is False
    r = _run_watcher(["--label", "p", "--probe", "echo 5",
                      "--pid", "99999999999999999999",
                      "--poll-seconds", "0.2", "--max-seconds", "1"])
    assert "Traceback" not in r.stderr, r.stderr
    assert _kinds(r.stdout), "the watcher must still say something"


def test_a_timed_out_probe_leaves_no_orphaned_grandchildren(mod):
    """subprocess's timeout kills only the direct /bin/sh child. Every process
    a PIPELINE forked -- including the exact probe shape this script's own help
    recommends -- used to survive, reparent to pid 1 and keep running: one leaked
    process per timed-out poll, forever, each doing whatever the probe's first
    stage does."""
    marker = "WJ_REGRESSION_ORPHAN_MARKER_4b7e"
    probe = (f'{sys.executable} -c "import time; time.sleep(60)  # {marker}"'
             f' | wc -l')
    for _ in range(3):
        value, err = mod.probe_value(probe, timeout=1.0)
        assert value is None and "timed out" in err

    survivors = []
    for _ in range(20):
        ps = subprocess.run(["ps", "-Ao", "pid,command"],
                            capture_output=True, text=True).stdout
        survivors = [l for l in ps.splitlines()
                     if marker in l and "ps -Ao" not in l]
        if not survivors:
            break
        time.sleep(0.25)
    # never leave debris behind, even if the assertion is about to fail
    for line in survivors:
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(line.split()[0]), signal.SIGKILL)
    assert not survivors, f"probe timeout leaked {len(survivors)} orphan(s)"


def test_the_probe_runs_in_its_own_session(mod):
    """The group-kill above is only safe because the probe is a session leader:
    the group we kill is one we created and is never our own."""
    out, err = mod.probe_value("ps -o pgid= -p $$ | tr -d ' '")
    assert err is None
    assert out != os.getpgid(0), "probe shares the watcher's process group"


def test_the_scan_of_a_huge_log_is_bounded(fresh, tmp_path):
    """scan_log used to read the ENTIRE unread portion into one string plus a
    full list of its lines -- a measured ~4x the file size, linear and
    unbounded, with the watcher frozen for the duration. Pointed at a large
    pre-existing log that is machine exhaustion caused BY the watcher."""
    log = tmp_path / "huge.log"
    line = b"2026-08-21 12:00:00 INFO worker step ok latency_ms=12 tokens=17\n"
    with log.open("wb") as fh:
        for _ in range(400):
            fh.write(line * 500)          # ~19 MB, well over the 4 MiB budget
    size = log.stat().st_size
    assert size > fresh.LOG_READ_BUDGET * 3

    scanner = fresh.LogScanner(log, 0, "big")
    t0 = time.time()
    first = scanner.scan()
    assert first - 0 <= fresh.LOG_READ_BUDGET, (
        f"one poll consumed {first} bytes; the per-poll budget is "
        f"{fresh.LOG_READ_BUDGET}")
    assert time.time() - t0 < 5, "a single poll must not freeze the watcher"
    # and it keeps making progress across polls rather than stalling forever
    assert scanner.scan() > first


def test_no_wall_clock_is_used_for_any_decision():
    """time.time() is CLOCK_REALTIME: adjustable, and it jumps when a laptop
    lid closes or NTP steps. Every duration and every stall decision must be
    monotonic."""
    src = SCRIPT.read_text()
    body = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#"))
    assert "time.time()" not in body, "wall clock leaked back into the logic"
    assert "time.monotonic()" in body


def test_a_clock_jump_does_not_fabricate_a_stall(fresh):
    """An 8-hour forward wall-clock jump on a HEALTHY job used to produce
    'STALL ... no progress for 28860s' and make --max-seconds fire hours early."""
    values = []
    for i in range(200):
        values.append(i // 4)             # advances every 4 polls: healthy
    out, code = _drive(fresh, ["--label", "nap", "--probe", "x",
                               "--stall-seconds", "300",
                               "--max-seconds", "3600",
                               "--heartbeat-seconds", "0"],
                       values, wall_jump_at=5, wall_jump=8 * 3600, poll=30.0)
    assert "STALL" not in _counts(out), out
    timeout_line = [l for l in out.splitlines() if l.startswith("TIMEOUT")][0]
    assert "8h" not in timeout_line and "28" not in timeout_line.split("after")[1][:6]


# ------------------------------------------------ priority 2: FALSE CALM
def test_an_oscillating_probe_still_stalls(fresh):
    """THE most likely real-world false calm. A temp/lock/partial file written
    then unlinked by a retry loop makes the probe flicker 41,42,41,42. Any
    CHANGE used to count as progress and reset the stall timer, so the watcher
    emitted PROGRESS forever over a job doing nothing."""
    values = [41, 42] * 200
    out, code = _drive(fresh, ["--label", "osc", "--probe", "x",
                               "--stall-seconds", "300",
                               "--max-seconds", "7200",
                               "--progress-every", "300",
                               "--heartbeat-seconds", "0"], values, poll=30.0)
    counts = _counts(out)
    assert counts.get("STALL", 0) >= 1, f"oscillation read as progress: {out}"
    assert "PROGRESS" not in counts or counts["PROGRESS"] <= 1, out


def test_a_probe_that_goes_backwards_is_not_progress(fresh):
    """--probe promises a number that GROWS. A decrease is informative (a cache
    being evicted, an output rewritten) but it is not progress and must not
    reset the stall timer."""
    values = [100] + list(range(99, 0, -1)) * 5
    out, code = _drive(fresh, ["--label", "d", "--probe", "x",
                               "--stall-seconds", "300",
                               "--max-seconds", "7200",
                               "--progress-every", "300",
                               "--heartbeat-seconds", "0"], values, poll=30.0)
    assert "STALL" in _counts(out), out
    assert "DROP" in _counts(out), "a decrease should be reported, not hidden"


def test_a_decreasing_probe_is_supported_when_asked_for(fresh):
    """Shrinking probes (queue depth, items remaining) are legitimate -- the
    fix must not delete that use, only stop it being the silent default."""
    out, code = _drive(fresh, ["--label", "q", "--probe", "x",
                               "--probe-direction", "down",
                               "--stall-seconds", "300",
                               "--max-seconds", "7200",
                               "--progress-every", "1",
                               "--heartbeat-seconds", "0"],
                       list(range(500, 0, -1)), poll=30.0)
    assert "PROGRESS" in _counts(out), out
    assert "STALL" not in _counts(out), out


def test_the_delta_is_rendered_readably(fresh):
    """It used to print '(+-1)' for a decrease."""
    out, _ = _drive(fresh, ["--label", "n", "--probe", "x",
                            "--probe-direction", "any",
                            "--stall-seconds", "300", "--max-seconds", "600",
                            "--progress-every", "1", "--heartbeat-seconds", "0"],
                    [1, 5, 9, 20], poll=30.0)
    assert "(+-" not in out and "(--" not in out, out
    assert re.search(r"\(\+\d", out), out


def test_a_stall_is_re_announced_and_never_latches_into_silence(fresh):
    """A job that dies at hour 1 of a 12-hour watch used to yield ONE line at
    minute 5 and nothing for the next eleven hours. The operator cannot tell
    that from 'watcher fine, job fine'."""
    out, code = _drive(fresh, ["--label", "dead", "--probe", "x",
                               "--stall-seconds", "300",
                               "--max-seconds", str(12 * 3600),
                               "--heartbeat-seconds", "0"],
                       [7] * 5000, poll=30.0)
    counts = _counts(out)
    assert counts.get("STALL", 0) >= 5, (
        f"12 hours of unbroken stall produced {counts.get('STALL', 0)} STALL "
        f"line(s); silence reads as calm")
    # ...but backoff keeps the total small enough not to trip Monitor's auto-stop
    assert counts["STALL"] < 40, f"too chatty: {counts}"
    assert "STILL stalled" in out


def test_the_watcher_emits_a_heartbeat_so_silence_means_it_is_dead(fresh):
    """The strongest anti-false-calm property: a line at least every
    --heartbeat-seconds, always. Then silence is never ambiguous."""
    out, code = _drive(fresh, ["--label", "hb", "--probe", "x",
                               "--stall-seconds", str(10 * 3600),
                               "--max-seconds", str(4 * 3600),
                               "--heartbeat-seconds", "900"],
                       [7] * 5000, poll=30.0)
    assert _counts(out).get("ALIVE", 0) >= 3, out
    assert "no news" in out


def test_sigterm_makes_the_watcher_announce_its_own_death():
    """SIGTERM is what Monitor's auto-stop, TaskStop, a kill, and a reboot all
    send -- the most likely way this watcher ends. It used to say nothing at
    all, so its own death was indistinguishable from a calm healthy job."""
    p = subprocess.Popen([sys.executable, str(SCRIPT), "--label", "sig",
                          "--probe", "echo 7", "--poll-seconds", "0.2",
                          "--stall-seconds", "60"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, cwd=str(REPO_ROOT))
    try:
        time.sleep(1.5)
        p.send_signal(signal.SIGTERM)
        out, err = p.communicate(timeout=20)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    assert "STOPPED" in _kinds(out), f"stdout={out!r} stderr={err!r}"
    assert "NOT BEING WATCHED" in out
    assert p.returncode == 130


def test_sigint_makes_the_watcher_announce_its_own_death():
    p = subprocess.Popen([sys.executable, str(SCRIPT), "--label", "sig",
                          "--probe", "echo 7", "--poll-seconds", "0.2",
                          "--stall-seconds", "60"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, cwd=str(REPO_ROOT))
    try:
        time.sleep(1.5)
        p.send_signal(signal.SIGINT)
        out, err = p.communicate(timeout=20)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    assert "STOPPED" in _kinds(out), f"stdout={out!r} stderr={err!r}"
    assert "KeyboardInterrupt" not in err, err


def test_an_unexpected_error_is_announced_rather_than_silent(fresh, capsys):
    """Under a driver that reads only stdout, a bare traceback on stderr is
    silence. The watcher must never end without saying so."""
    def boom(argv=None):
        raise RuntimeError("something nobody predicted")
    fresh.main = boom
    code = fresh.run(["--label", "kaboom", "--probe", "echo 1"])
    out = capsys.readouterr().out
    assert "WATCHER-DIED" in out and "kaboom" in out, out
    assert "NO LONGER BEING WATCHED" in out
    assert code == fresh.EXIT_WATCHER_DIED


def test_a_zombie_process_is_not_alive(mod):
    """os.kill(pid, 0) succeeds for a process that exited but was never reaped.
    The watched job is FINISHED -- possibly crashed -- and the watcher used to
    report process=alive and never emit EXIT. That is this file's founding
    incident, exactly."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])   # deliberately unreaped
    try:
        for _ in range(40):
            state = subprocess.run(["ps", "-o", "state=", "-p", str(p.pid)],
                                   capture_output=True, text=True).stdout.strip()
            if state.startswith("Z"):
                break
            time.sleep(0.05)
        assert state.startswith("Z"), f"could not create a zombie (state={state!r})"
        assert mod.pid_alive(p.pid) is False, "a defunct process read as alive"
    finally:
        p.wait()


def test_a_watch_is_bounded_by_default():
    """--pid, --target and --max-seconds were all optional, so the default
    invocation had NO exit condition. Copied into a global config, those
    accumulate as immortal orphans across every project and session."""
    parser_default = None
    for line in subprocess.run([sys.executable, str(SCRIPT), "--help"],
                               capture_output=True, text=True).stdout.splitlines():
        if "--max-seconds" in line or "default 86400" in line:
            parser_default = (parser_default or "") + line
    assert parser_default and "86400" in parser_default

    p = subprocess.Popen([sys.executable, str(SCRIPT), "--label", "orphan",
                          "--probe", "echo 7", "--poll-seconds", "0.2",
                          "--stall-seconds", "0.5", "--max-seconds", "2"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, cwd=str(REPO_ROOT))
    try:
        out, _ = p.communicate(timeout=20)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    assert p.returncode is not None and "TIMEOUT" in _kinds(out)


# --------------------------------------------- priority 2: FALSE SUCCESS
def test_the_probe_output_must_be_a_number_not_merely_contain_one(mod):
    """re.search(r'-?\\d+') took the FIRST digit run anywhere in stdout. With
    --target that produced an immediate, confident, false TARGET and exit 0 --
    a positive signal a wrapper script branches on, for a job that did
    nothing."""
    cases = {
        "echo '2026-08-21 12:00:00 done=3'": "read the YEAR",
        "echo 'runs/2026/out.jsonl'": "read a path component",
        "echo 'found 0 files in 3 dirs'": "read the preamble",
        "echo '1,234,567'": "read only the first group",
        "echo 'sh-3.2$ 55'": "read a NEGATIVE number out of a prompt",
        "echo 'Python 3.11 warning'; echo 812": "read the version",
    }
    for cmd, why in cases.items():
        value, err = mod.probe_value(cmd)
        assert value is None, f"{cmd!r} still guesses a number ({why})"
        assert "no number" in err


def test_a_plain_number_is_still_accepted_in_every_ordinary_shape(mod):
    assert mod.probe_value("echo 42")[0] == 42
    assert mod.probe_value("printf '  7\\n'")[0] == 7
    assert mod.probe_value("printf '5\\n\\n'")[0] == 5
    assert mod.probe_value("echo 0")[0] == 0
    assert mod.probe_value("echo 3.5")[0] == 3.5
    assert mod.probe_value("echo -12")[0] == -12


def test_permissive_parsing_is_available_but_must_be_asked_for(mod):
    """Do not delete the convenience -- just stop guessing by default."""
    assert mod.probe_value("echo 'found 3 files'", parse="last")[0] == 3
    assert mod.probe_value("echo 'found 3 files'")[0] is None


def test_an_absurd_digit_count_is_a_probe_fail_not_a_crash(mod):
    """int() raises ValueError past 4300 digits. It fired on the FIRST probe,
    before START was even printed, and killed the watch with a traceback."""
    value, err = mod.probe_value(f"{sys.executable} -c \"print('9'*4400)\"")
    assert value is None and "no number" in err


def test_a_pipeline_probe_cannot_hide_its_producers_failure(mod):
    """A shell pipeline exits with the status of its LAST stage, so
    `find missing/ -type f | wc -l` exits 0 and prints 0. The watcher then
    reported a confident probe=0 forever and sent the operator to debug the job
    instead of the probe. This is the shape the script's own help recommends,
    so a wrong path is the MOST likely input in a global install."""
    value, err = mod.probe_value("find /definitely/not/here -type f | wc -l")
    assert value is None, "a blind probe still reads as a healthy 0"
    assert "exited" in err

    r = _run_watcher(["--label", "k", "--probe",
                      "find /definitely/not/here -type f | wc -l",
                      "--poll-seconds", "0.2", "--stall-seconds", "1",
                      "--max-seconds", "1.5"])
    assert "PROBE-FAIL" in _kinds(r.stdout), r.stdout


def test_a_target_already_satisfied_at_start_is_refused(tmp_path):
    """FALSE SUCCESS. A watcher pointed at a stale output file from a previous
    run emitted TARGET and exit 0 within one poll -- certifying success for a
    job that produced nothing and was already dead."""
    stale = tmp_path / "out.jsonl"
    stale.write_text("x\n" * 147)
    r = _run_watcher(["--label", "eval", "--probe", f"wc -l < {stale}",
                      "--target", "147", "--poll-seconds", "0.2",
                      "--max-seconds", "3"])
    assert "TARGET-PRESET" in _kinds(r.stdout), r.stdout
    assert r.returncode != 0, "exit 0 certifies a success nobody observed"
    assert r.returncode == 3


def test_a_preexisting_target_can_be_accepted_deliberately(tmp_path):
    """The escape hatch exists -- it just has to be asked for, awake."""
    stale = tmp_path / "out.jsonl"
    stale.write_text("x\n" * 147)
    r = _run_watcher(["--label", "eval", "--probe", f"wc -l < {stale}",
                      "--target", "147", "--target-allow-preexisting",
                      "--poll-seconds", "0.2", "--max-seconds", "3"])
    assert "TARGET" in _kinds(r.stdout) and r.returncode == 0, r.stdout


def test_a_target_met_by_an_already_dead_process_says_so(tmp_path):
    stale = tmp_path / "out.jsonl"
    stale.write_text("x\n" * 10)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    r = _run_watcher(["--label", "e", "--probe", f"wc -l < {stale}",
                      "--target", "10", "--target-allow-preexisting",
                      "--pid", str(dead.pid), "--poll-seconds", "0.2",
                      "--max-seconds", "3"])
    assert r.returncode == 0
    assert "TARGET" in _kinds(r.stdout)


# ------------------------------------------------- priority 2: SELF-INFLICTED NOISE
def test_probe_failures_are_throttled_and_eventually_terminal(fresh):
    """A typo'd probe, or one pointing at a path the job has not created yet,
    emitted one PROBE-FAIL per poll forever. At the default 30s poll a 12-hour
    watch is ~1440 identical notifications -- and Monitor stops a source that
    produces too many events, so the flood ends in silence."""
    out, code = _drive(fresh, ["--label", "typo", "--probe", "x",
                               "--stall-seconds", "300",
                               "--max-seconds", str(12 * 3600),
                               "--probe-fail-limit", "0",
                               "--heartbeat-seconds", "0"],
                       ["probe exited 1: no such file"] * 5000, poll=30.0)
    n = _counts(out).get("PROBE-FAIL", 0)
    assert 1 <= n <= 30, f"{n} PROBE-FAIL lines over 12 hours is a flood"
    assert "in a row" in out


def test_a_permanently_broken_probe_gives_up_loudly(fresh):
    """A watcher blind for 20 polls is not watching anything, and must say so
    instead of looping forever."""
    out, code = _drive(fresh, ["--label", "blind", "--probe", "x",
                               "--stall-seconds", "300",
                               "--max-seconds", str(12 * 3600),
                               "--probe-fail-limit", "20",
                               "--heartbeat-seconds", "0"],
                       ["probe exited 1: nope"] * 5000, poll=30.0)
    assert "PROBE-DEAD" in _counts(out), out
    assert code == fresh.EXIT_PROBE_DEAD
    assert "BLIND, not calm" in out


def test_a_probe_that_recovers_says_so(fresh):
    out, code = _drive(fresh, ["--label", "r", "--probe", "x",
                               "--stall-seconds", "300", "--max-seconds", "600",
                               "--heartbeat-seconds", "0"],
                       ["probe exited 1: nope", "probe exited 1: nope", 5, 6, 7],
                       poll=30.0)
    assert "PROBE-OK" in _counts(out), out


def test_multiline_probe_stderr_stays_exactly_one_event(tmp_path):
    """A probe whose stderr is a traceback turned ONE failure into three
    Monitor events, and the extra lines' first tokens became fabricated event
    kinds."""
    bad = tmp_path / "bad.py"
    bad.write_text("import sys\n"
                   "sys.stderr.write('Traceback (most recent call last):\\n"
                   "  File \"x.py\", line 1\\nEXIT      [x] all good\\n')\n"
                   "sys.exit(1)\n")
    r = _run_watcher(["--label", "real", "--probe", f"{sys.executable} {bad}",
                      "--poll-seconds", "0.3", "--max-seconds", "0.4"])
    kinds = set(_kinds(r.stdout))
    assert kinds <= {"START", "PROBE-FAIL", "TIMEOUT"}, (
        f"probe stderr forged event kinds: {kinds}")
    assert "EXIT" not in kinds


def test_a_newline_in_the_label_cannot_forge_an_event():
    """--label in a global config is routinely interpolated from $(basename
    $PWD), a branch name or a commit subject. A newline in it used to split
    every event into several, the extras' first tokens becoming invented event
    kinds."""
    r = _run_watcher(["--label", "job\nSTALL     [prod] disk full, killing everything",
                      "--probe", "echo 1", "--poll-seconds", "0.3",
                      "--max-seconds", "0.4"])
    kinds = set(_kinds(r.stdout))
    assert "STALL" not in kinds, f"forged a STALL out of a label: {r.stdout!r}"
    assert kinds <= {"START", "TIMEOUT"}, kinds


def test_every_event_is_exactly_one_line(mod, capsys):
    mod.emit("START", "a\nb\rc", "one\ntwo\nthree")
    out = capsys.readouterr().out
    assert out.count("\n") == 1, repr(out)


# ------------------------------------------------------- priority 2: the log
def test_benign_log_lines_do_not_raise_alarms(fresh, tmp_path):
    """Bare substrings matched 'Error' inside 'train_error=', 'OOM' inside
    'Zooming', '429' inside 'step 4290/10000'. On realistic healthy output from
    an LLM training project the false-positive rate was 100%, which teaches the
    reader to ignore LOGFAIL -- so the one real Traceback is ignored too."""
    benign = [
        "epoch 3 | loss 0.214 | error_rate 0.000",
        "Zooming in on the training curve",
        "wrote 4290 rows to shard-00",
        "Bloomberg feed connected",
        "quota check: OK, 98% headroom remaining",
        "no exceptions raised",
        "GET /v1/messages 200 OK in 429ms",
        "epoch 1: train_error=0.0031 val_error=0.0029",
        "Terror Management Theory corpus loaded",
        "INFO Killed 0 stale workers",
        "PASSED tests/test_error_handling.py::test_no_errors",
        "0 errors, 429 items processed",
        "Loss: 0.0429 | step 4290/10000",
        "commit 429abc1 checked out",
        "groomed the index",
    ]
    log = tmp_path / "healthy.log"
    log.write_text("\n".join(benign) + "\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fresh.scan_log(log, 0, "healthy")
    fired = [l for l in buf.getvalue().splitlines() if l.startswith("LOGFAIL")]
    assert not fired, "false alarms on healthy lines:\n" + "\n".join(fired)


def test_real_failures_are_still_caught(fresh, tmp_path):
    """The point of tightening the patterns is NOT to go quiet."""
    real = [
        "Traceback (most recent call last):",
        "RuntimeError: the job died",
        "ValueError: bad shape",
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate",
        "Killed",
        "FAILED tests/test_x.py::test_y",
        "anthropic.RateLimitError: rate limited",
        "HTTP 429 Too Many Requests",
        "quota exceeded for project",
        "fatal: not a git repository",
    ]
    for line in real:
        log = tmp_path / "r.log"
        log.write_text(line + "\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fresh.scan_log(log, 0, "x")
        assert "LOGFAIL" in buf.getvalue(), f"missed a real failure: {line!r}"


def test_a_rotated_log_does_not_swallow_the_crash(fresh, tmp_path):
    """The truncation guard only fired when the replacement was SMALLER than
    the stored offset. A log rotated and regrown past the old offset -- the
    normal case for a job that logs steadily -- had everything before that
    offset silently skipped, INCLUDING the crash that caused the rotation."""
    log = tmp_path / "rot.log"
    log.write_text("".join(f"line {i} all fine here\n" for i in range(30)))
    scanner = fresh.LogScanner(log, 0, "x")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        offset = scanner.scan()
    assert offset > 0 and "LOGFAIL" not in buf.getvalue()

    log.unlink()                                   # rotate: brand new inode
    log.write_text("Traceback (most recent call last):\n"
                   "RuntimeError: the job died\n"
                   + "".join(f"filler {i}\n" for i in range(60)))
    assert log.stat().st_size > offset, "setup must make the NEW file bigger"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        scanner.scan()
    assert "RuntimeError: the job died" in buf.getvalue(), (
        "the crash that caused the rotation was skipped:\n" + buf.getvalue())
    assert "LOGROTATE" in buf.getvalue()


def test_a_truncated_and_regrown_log_is_rescanned(fresh, tmp_path):
    """Same inode, reopened with '>' -- st_size can already be past the old
    offset by the next poll."""
    log = tmp_path / "t.log"
    log.write_text("".join(f"healthy line {i}\n" for i in range(40)))
    scanner = fresh.LogScanner(log, 0, "x")
    with contextlib.redirect_stdout(io.StringIO()):
        offset = scanner.scan()
    with log.open("w") as fh:                       # truncate in place
        fh.write("Traceback (most recent call last):\n")
        fh.write("RuntimeError: died after rotation\n")
        for i in range(60):
            fh.write(f"filler filler filler {i}\n")
    assert log.stat().st_size > offset
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        scanner.scan()
    assert "died after rotation" in buf.getvalue(), buf.getvalue()


def test_a_growing_log_is_not_mistaken_for_a_rotated_one(fresh, tmp_path):
    """The identity fingerprint must not false-positive on a log that was
    simply short at the first poll."""
    log = tmp_path / "g.log"
    log.write_text("short\n")
    scanner = fresh.LogScanner(log, 0, "x")
    with contextlib.redirect_stdout(io.StringIO()):
        scanner.scan()
    with log.open("a") as fh:
        fh.write("x" * 500 + "\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        scanner.scan()
    assert "LOGROTATE" not in buf.getvalue(), buf.getvalue()


def test_an_existing_log_is_not_replayed_as_a_flood(tmp_path):
    """log_offset started at 0, so arming --log on a job with an existing log
    replayed every historical match at once -- 500 events on the FIRST poll in
    the reproduction."""
    log = tmp_path / "old.log"
    log.write_text("".join(f"RuntimeError: old failure {i}\n" for i in range(500)))
    r = _run_watcher(["--label", "pre", "--probe", "echo 1", "--log", str(log),
                      "--poll-seconds", "0.3", "--max-seconds", "0.4"])
    assert _kinds(r.stdout).count("LOGFAIL") == 0, r.stdout


def test_the_history_of_a_log_can_still_be_scanned_on_request(tmp_path):
    log = tmp_path / "old.log"
    log.write_text("".join(f"RuntimeError: old failure {i}\n" for i in range(20)))
    r = _run_watcher(["--label", "pre", "--probe", "echo 1", "--log", str(log),
                      "--log-from-start", "--poll-seconds", "0.3",
                      "--max-seconds", "0.4"])
    assert "LOGFAIL" in _kinds(r.stdout), r.stdout


def test_no_single_poll_can_flood_the_event_stream(fresh, tmp_path):
    """Monitor stops a source that produces too many events, and a stopped
    watcher is a silent one."""
    log = tmp_path / "many.log"
    log.write_text("".join(f"RuntimeError: distinct failure {i}\n"
                           for i in range(4000)))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fresh.scan_log(log, 0, "x")
    n = buf.getvalue().count("LOGFAIL")
    assert n <= fresh.LOG_EVENTS_PER_POLL + 1, f"{n} events from one poll"
    assert "more matching lines this poll" in buf.getvalue()


def test_identical_log_lines_are_not_reported_over_and_over(fresh, tmp_path):
    log = tmp_path / "rep.log"
    scanner = fresh.LogScanner(log, 0, "x")
    for _ in range(6):
        with log.open("a") as fh:
            fh.write("RuntimeError: the same failure every time\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            scanner.scan()
        counts = buf.getvalue().count("LOGFAIL")
    log.write_text(log.read_text())
    assert counts == 0, "the same line was re-notified on a later poll"


def test_a_log_that_is_a_directory_is_refused_not_fatal(fresh, tmp_path):
    """path.stat() succeeds for a directory, so the OSError guard never fired
    and open() raised IsADirectoryError on poll 1 -- START printed, then dead."""
    assert fresh.scan_log(tmp_path, 0, "x") == 0        # must not raise
    r = _run_watcher(["--label", "dirlog", "--probe", "echo 1",
                      "--log", str(tmp_path), "--poll-seconds", "0.3",
                      "--max-seconds", "1"])
    assert r.returncode == 2, r.stdout
    assert "regular file" in r.stderr
    assert "Traceback" not in r.stderr


# ------------------------------------------------------- priority 4: clarity
def test_a_broken_pipe_does_not_produce_a_traceback(tmp_path):
    """An agent piping this into `head`, or a terminal that closes, used to end
    the watcher with a two-part BrokenPipeError traceback.

    The probe genuinely GROWS (one byte per poll) so the watcher keeps emitting
    past `head -2` and really does hit EPIPE -- a test that merely fails to
    produce output would pass vacuously.
    """
    counter = tmp_path / "c"
    counter.write_text("")
    probe = f"printf x >> {counter}; wc -c < {counter}"
    # stderr must NOT be folded into the pipe: `2>&1 | head` would send the
    # traceback into the very pipe that is closed, and the test could never
    # see it.
    cmd = (f"{sys.executable} {SCRIPT} --label f --probe {probe!r} "
           f"--poll-seconds 0.05 --progress-every 0.01 --max-seconds 5 "
           f"| head -2")
    r = subprocess.run(["/bin/bash", "-c", cmd], capture_output=True, text=True,
                       timeout=40, cwd=str(REPO_ROOT))
    assert len(r.stdout.splitlines()) == 2, (
        f"the pipe was never actually broken; got {r.stdout!r}")
    assert counter.stat().st_size > 2, "the watcher never polled past the pipe close"
    assert "BrokenPipeError" not in r.stderr, r.stderr
    assert "Traceback" not in r.stderr, r.stderr


def test_help_teaches_an_agent_that_has_never_seen_this_script():
    """The reader is usually an agent skimming --help before writing a probe
    for a project it does not know. If --help does not say what a good probe
    is, the agent picks one of the shapes that breaks this thing."""
    r = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                       capture_output=True, text=True, timeout=30)
    h = r.stdout
    assert r.returncode == 0
    for kind in ("START", "PROGRESS", "STALL", "RECOVER", "EXIT", "TARGET",
                 "PROBE-FAIL", "PROBE-DEAD", "LOGFAIL", "TIMEOUT", "STOPPED",
                 "WATCHER-DIED", "ALIVE", "DROP"):
        assert kind in h, f"--help never mentions the {kind} event"
    assert "EXIT CODES" in h
    for code in ("0", "1", "2", "3", "4", "5", "130"):
        assert re.search(rf"^\s+{code}\s+\S", h, re.M), f"exit code {code} undocumented"
    assert "ONE number" in h and "GROW" in h
    assert "pipefail" in h, "--help must warn that a pipeline hides its error"
    assert "buffers stdout" in h, "--help must warn against probing the job's log"
    assert "--poll-seconds" in h
    poll_help = h.split("--poll-seconds")[-1]
    assert "how often" in poll_help.lower()
    assert "SILENCE" in h


def test_the_script_is_stdlib_only_and_runs_on_old_pythons():
    """It is going into a global config and will run in unknown projects, under
    whatever interpreter is around."""
    import ast
    tree = ast.parse(SCRIPT.read_text())
    # every one of these ships with CPython 3.8; `selectors` is 3.4+
    stdlib = {"argparse", "os", "re", "selectors", "shutil", "signal",
              "subprocess", "sys", "time", "collections", "pathlib", "typing",
              "errno", "ast"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] in stdlib, f"non-stdlib import {a.name}"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            root = node.module.split(".")[0]
            assert root in stdlib or root == "__future__", \
                f"non-stdlib import {node.module}"
    src = SCRIPT.read_text()
    # The real check: does this file PARSE as Python 3.8? Beats grepping for
    # individual newer constructs, which only catches the ones you thought of.
    ast.parse(src, feature_version=(3, 8))
    assert "removeprefix" not in src and "removesuffix" not in src   # 3.9+ API
    assert "BooleanOptionalAction" not in src                        # 3.9+ API
    assert "/Users/" not in src and "/home/" not in src, "hardcoded path"


def test_durations_are_never_rounded_away_to_zero():
    """START used to print 'poll=0s' for --poll-seconds 0.3."""
    r = _run_watcher(["--label", "fmt", "--probe", "echo 1",
                      "--poll-seconds", "0.3", "--max-seconds", "0.4"])
    start = [l for l in r.stdout.splitlines() if l.startswith("START")][0]
    assert "poll=0s" not in start, start


def test_the_start_line_can_never_hide_which_pid_is_watched():
    """`if args.pid` is falsy for 0, so --pid 0 produced a START line identical
    to a no-pid watch."""
    src = SCRIPT.read_text()
    assert "if args.pid is not None" in src
    assert not re.search(r'if args\.pid\b(?!\s+is)', src), \
        "a truthiness test on --pid can hide pid=0"


def test_the_heartbeat_never_reads_as_reassuring_during_a_stall(fresh):
    """A heartbeat that says 'no news' while the job is dead would reintroduce
    the very false calm it exists to prevent."""
    out, _ = _drive(fresh, ["--label", "hb", "--probe", "x",
                            "--stall-seconds", "300",
                            "--max-seconds", str(6 * 3600),
                            "--heartbeat-seconds", "900"],
                    [7] * 5000, poll=30.0)
    alive = [l for l in out.splitlines() if l.startswith("ALIVE")]
    assert alive, out
    assert all("STILL STALLED" in l for l in alive), (
        "a heartbeat said 'no news' while the job was stalled:\n"
        + "\n".join(alive[:3]))


def test_the_plain_scan_log_helper_is_rotation_safe_too(fresh, tmp_path):
    """The exported scan_log(path, offset, label) helper is the shape a caller
    naturally reaches for in a poll loop. If it forgets the file's identity
    between calls it seeks into the replacement and skips the crash -- the same
    defect as the watcher's, just one layer out."""
    log = tmp_path / "rot2.log"
    log.write_text("".join(f"benign line {i}\n" for i in range(20)))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        offset = fresh.scan_log(log, 0, "x")
    assert offset > 0 and "LOGFAIL" not in buf.getvalue()

    log.unlink()
    log.write_text("Traceback (most recent call last):\n"
                   "RuntimeError: the job died\n"
                   + "".join(f"filler {i}\n" for i in range(40)))
    assert log.stat().st_size > offset
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fresh.scan_log(log, offset, "x")
    assert "the job died" in buf.getvalue(), (
        "scan_log skipped the crash after rotation:\n" + repr(buf.getvalue()))


def test_the_scanner_cache_cannot_grow_without_bound(fresh, tmp_path):
    for i in range(fresh._SCANNER_MEMORY * 3):
        p = tmp_path / f"log{i}.log"
        p.write_text("fine\n")
        fresh.scan_log(p, 0, "x")
    assert len(fresh._SCANNERS) <= fresh._SCANNER_MEMORY


def test_a_short_timeout_is_reported_accurately(mod):
    """'timed out after 2s' for a 1.5s timeout is a small lie in a message the
    operator uses to size the next timeout."""
    value, err = mod.probe_value("sleep 5", timeout=1.5)
    assert value is None
    assert "1.5s" in err, err


def test_a_runaway_probe_cannot_exhaust_memory(mod):
    """subprocess.communicate() has no size limit. `--probe "cat out.jsonl"` is
    a plausible slip for `wc -l < out.jsonl`, and it pulled the whole file into
    the watcher's RAM -- measured 1332MB for a 400MB probe. A watcher must never
    be the thing that exhausts the machine."""
    import resource
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    value, err = mod.probe_value(
        "yes 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' | head -c 200000000",
        timeout=30)
    grew = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - base)
    assert value is None and "more than" in err, (value, err)
    assert grew < 100 * 1024 * 1024, f"probe output grew RSS by {grew} bytes"


def test_a_runaway_probe_does_not_hang_the_watcher(mod):
    """Output past the cap must still be DRAINED, or the child wedges on a full
    pipe and the bounded read becomes an unbounded wait."""
    t0 = time.time()
    value, err = mod.probe_value("yes hello", timeout=6)
    assert time.time() - t0 < 15, "a probe that never stops output wedged us"
    assert value is None


def test_ordinary_probe_output_is_untouched_by_the_cap(mod):
    assert mod.probe_value("echo 12345")[0] == 12345


def test_it_actually_runs_under_the_oldest_python_available():
    """Parsing as 3.8 is necessary but not sufficient -- `{} | {}` and friends
    are runtime errors, not syntax errors. So really run it somewhere older
    than the dev interpreter. Skips cleanly if this machine has only one."""
    import shutil as _shutil
    here = sys.version_info[:2]
    candidates = []
    for name in ("python3.8", "python3.9", "python3.10", "python3.11", "python3"):
        path = _shutil.which(name) or f"/usr/bin/{name}"
        if not os.path.exists(path):
            continue
        try:
            v = subprocess.run([path, "-c",
                                "import sys;print('%d.%d'%sys.version_info[:2])"],
                               capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if v.returncode != 0:
            continue
        got = tuple(int(x) for x in v.stdout.strip().split("."))
        if got < here:
            candidates.append((got, path))
    if not candidates:
        pytest.skip("no interpreter older than the current one on this machine")
    version, python = sorted(candidates)[0]

    probe = f"{python} -c \"print(7)\""
    r = subprocess.run([python, str(SCRIPT), "--label", "old", "--probe", probe,
                        "--poll-seconds", "0.2", "--stall-seconds", "0.5",
                        "--max-seconds", "1.2"],
                       capture_output=True, text=True, timeout=60,
                       cwd=str(REPO_ROOT))
    assert "Traceback" not in r.stderr, (
        f"failed under python {version[0]}.{version[1]}:\n{r.stderr}")
    kinds = _kinds(r.stdout)
    assert "START" in kinds and "STALL" in kinds and "TIMEOUT" in kinds, r.stdout


def test_no_python_39_plus_runtime_apis_are_used():
    """ast.parse(feature_version=(3, 8)) cannot see these -- `{} | {}` is a
    TypeError at run time on 3.8, not a SyntaxError."""
    src = SCRIPT.read_text()
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert not re.search(r"\}\s*\|\s*\{", body), "dict | merge is 3.9+"
    assert "str.removeprefix" not in body and ".removesuffix(" not in body
    assert "functools.cache" not in body                       # 3.9+
    assert "graphlib" not in body                              # 3.9+
    assert "zoneinfo" not in body                              # 3.9+


def test_a_flapping_probe_does_not_flood(fresh):
    """Throttling only the FAILURES leaves a probe that flaps -- fail, ok,
    fail, ok -- emitting a PROBE-FAIL and a PROBE-OK every single poll forever.
    An output file rewritten atomically, or a resource that is intermittently
    there, produces exactly this. The probe-health channel has to be rate
    limited as ONE stream."""
    values = []
    for i in range(3000):
        values.append("probe exited 1: gone" if i % 2 else 5)
    out, _ = _drive(fresh, ["--label", "flap", "--probe", "x",
                            "--stall-seconds", "300",
                            "--max-seconds", str(12 * 3600),
                            "--probe-fail-limit", "0",
                            "--heartbeat-seconds", "0"], values, poll=30.0)
    counts = _counts(out)
    health = counts.get("PROBE-FAIL", 0) + counts.get("PROBE-OK", 0)
    assert health <= 40, f"flapping probe emitted {health} health events: {counts}"
    assert health >= 1, "a flapping probe must still be reported at all"
    assert "suppressed" in out, "the suppressed flapping must be visible"
    # and the flapping value never advances, so it must still STALL
    assert counts.get("STALL", 0) >= 1, counts


def test_a_new_failure_is_still_reported_promptly_after_a_quiet_spell(fresh):
    """Backoff must not leave a fresh problem waiting out an old timer."""
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + ["probe exited 1: broke"] * 20
    out, _ = _drive(fresh, ["--label", "q", "--probe", "x",
                            "--stall-seconds", "3000",
                            "--max-seconds", str(3600),
                            "--probe-fail-limit", "0",
                            "--heartbeat-seconds", "0"], values, poll=30.0)
    lines = out.splitlines()
    first_fail = next(i for i, l in enumerate(lines) if l.startswith("PROBE-FAIL"))
    # the probe broke on poll 11; the report must not lag far behind it
    assert first_fail <= 4, f"a new failure was reported late:\n{out}"
