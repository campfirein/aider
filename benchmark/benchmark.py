#!/usr/bin/env python3
import datetime
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from json.decoder import JSONDecodeError
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import git
import importlib_resources
import lox
import pandas as pd
import prompts
import typer
from dotenv import load_dotenv
from plots import plot_refactoring
from rich.console import Console

# Import ByteRover service
from byterover_service import ByteroverService

from aider import models, sendchat
from aider.coders import Coder, base_coder
from aider.dump import dump  # noqa: F401
from aider.io import InputOutput

BENCHMARK_DNAME = Path(os.environ.get("AIDER_BENCHMARK_DIR", "byterover.benchmarks"))

EXERCISES_DIR_DEFAULT = "polyglot-benchmark"

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


load_dotenv(override=True)


def find_latest_benchmark_dir():
    benchmark_dirs = [d for d in BENCHMARK_DNAME.iterdir() if d.is_dir()]
    if not benchmark_dirs:
        print("Error: No benchmark directories found under byterover.benchmarks.")
        sys.exit(1)

    # Get current time and 24 hours ago
    now = datetime.datetime.now()
    day_ago = now - datetime.timedelta(days=1)

    # Filter directories by name pattern YYYY-MM-DD-HH-MM-SS--
    recent_dirs = []
    for d in benchmark_dirs:
        try:
            # Extract datetime from directory name
            date_str = d.name[:19]  # Takes YYYY-MM-DD-HH-MM-SS
            dir_date = datetime.datetime.strptime(date_str, "%Y-%m-%d-%H-%M-%S")
            if dir_date >= day_ago:
                recent_dirs.append(d)
        except ValueError:
            # Skip directories that don't match the expected format
            continue

    if not recent_dirs:
        print("Error: No benchmark directories found from the last 24 hours.")
        sys.exit(1)

    # Find directory with most recently modified .md file
    latest_dir = None
    latest_time = 0

    for d in recent_dirs:
        # Look for .md files in subdirectories
        for md_file in d.glob("*/exercises/practice/*/.*.md"):
            if md_file.is_file():
                mtime = md_file.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_dir = d

    if not latest_dir:
        print("Error: No .md files found in recent benchmark directories.")
        sys.exit(1)

    print(f"Using the most recently updated benchmark directory: {latest_dir.name}")
    return latest_dir


def show_stats(dirnames, graphs, stats_languages=None):
    raw_rows = []
    for dirname in dirnames:
        row = summarize_results(dirname, stats_languages)
        raw_rows.append(row)

    # return

    seen = dict()
    rows = []
    for row in raw_rows:
        if not row:
            continue

        if row.completed_tests != row.total_tests:
            print(
                f"Warning: {row.dir_name} is incomplete: {row.completed_tests} of {row.total_tests}"
            )

        try:
            kind = (row.model, row.edit_format)
        except AttributeError:
            return

        if kind in seen:
            dump(row.dir_name)
            dump(seen[kind])
            return

        seen[kind] = row.dir_name
        rows.append(vars(row))

    repeat_hi = repeat_lo = repeat_avg = None  # noqa: F841

    df = pd.DataFrame.from_records(rows)
    # df.sort_values(by=["model", "edit_format"], inplace=True)

    # dump(df)
    if graphs:
        # plot_timing(df)
        # plot_outcomes(df, repeats, repeat_hi, repeat_lo, repeat_avg)
        # plot_outcomes_claude(df)
        plot_refactoring(df)


def resolve_dirname(dirname, use_single_prior, make_new):
    if len(dirname.parts) > 1:
        return dirname

    priors = list(BENCHMARK_DNAME.glob(f"*--{dirname}"))
    if len(priors) == 1 and use_single_prior:
        dirname = priors[0].name
        print(f"Using pre-existing {dirname}")
    elif len(priors):
        if not make_new:
            print(f"Prior runs of {dirname} exist, use --new or name one explicitly")
            print()
            for prior in priors:
                print(prior)
            return

    if not re.match(r"\d\d\d\d-\d\d-\d\d-", str(dirname)):
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S--")
        dirname = now + dirname.name

    dirname = BENCHMARK_DNAME / dirname
    return dirname


@app.command()
def main(
    dirnames: Optional[List[str]] = typer.Argument(None, help="Directory names"),
    graphs: bool = typer.Option(False, "--graphs", help="Generate graphs"),
    model: str = typer.Option("gpt-3.5-turbo", "--model", "-m", help="Model name"),
    sleep: float = typer.Option(
        0, "--sleep", help="Sleep seconds between tests when single threaded"
    ),
    languages: str = typer.Option(
        None, "--languages", "-l", help="Only run tests for specific languages (comma separated)"
    ),
    edit_format: str = typer.Option(None, "--edit-format", "-e", help="Edit format"),
    editor_model: str = typer.Option(None, "--editor-model", help="Editor model name"),
    editor_edit_format: str = typer.Option(None, "--editor-edit-format", help="Editor edit format"),
    replay: str = typer.Option(
        None,
        "--replay",
        help="Replay previous .aider.chat.history.md responses from previous benchmark run",
    ),
    keywords: str = typer.Option(
        None, "--keywords", "-k", help="Only run tests that contain keywords (comma sep)"
    ),
    clean: bool = typer.Option(
        False, "--clean", "-c", help="Discard the existing testdir and make a clean copy"
    ),
    cont: bool = typer.Option(False, "--cont", help="Continue the (single) matching testdir"),
    make_new: bool = typer.Option(False, "--new", "-n", help="Make a new dated testdir"),
    no_unit_tests: bool = typer.Option(False, "--no-unit-tests", help="Do not run unit tests"),
    no_aider: bool = typer.Option(False, "--no-aider", help="Do not run aider"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    stats_only: bool = typer.Option(
        False, "--stats", "-s", help="Do not run tests, just collect stats on completed tests"
    ),
    stats_languages: str = typer.Option(
        None,
        "--stats-languages",
        help="Only include stats for specific languages (comma separated)",
    ),
    diffs_only: bool = typer.Option(False, "--diffs", help="Just diff the provided stats dirs"),
    tries: int = typer.Option(2, "--tries", "-r", help="Number of tries for running tests"),
    threads: int = typer.Option(1, "--threads", "-t", help="Number of threads to run in parallel"),
    num_tests: int = typer.Option(-1, "--num-tests", help="Number of tests to run"),
    percentage: float = typer.Option(
        5.0,
        "--percentage",
        "-p",
        help="Percentage of benchmark tests to run (1-100), defaults to 5%. Use 100 for all tests.",
        min=1.0,
        max=100.0
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for test selection when using percentage"
    ),
    num_ctx: Optional[int] = typer.Option(
        None, "--num-ctx", help="Override model context window size"
    ),
    read_model_settings: str = typer.Option(
        None, "--read-model-settings", help="Load aider model settings from YAML file"
    ),
    reasoning_effort: Optional[str] = typer.Option(
        None, "--reasoning-effort", help="Set reasoning effort for models that support it"
    ),
    thinking_tokens: Optional[int] = typer.Option(
        None, "--thinking-tokens", help="Set thinking tokens for models that support it"
    ),
    exercises_dir: str = typer.Option(
        EXERCISES_DIR_DEFAULT, "--exercises-dir", help="Directory with exercise files"
    ),
    # ByteRover memory system options
    memory_create: bool = typer.Option(
        False, "--memory-create", help="Enable creation of memories from benchmark runs"
    ),
    memory_retrieve: bool = typer.Option(
        False, "--memory-retrieve", help="Enable retrieval of relevant memories for tasks"
    ),
    byterover_api_key: str = typer.Option(
        None, "--byterover-api-key", help="ByteRover API key", envvar="BYTEROVER_API_KEY"
    ),
    byterover_user_id: str = typer.Option(
        None, "--byterover-user-id", help="ByteRover User ID", envvar="BYTEROVER_USER_ID"
    ),
    memory_limit: int = typer.Option(
        3, "--memory-limit", help="Maximum number of memories to retrieve"
    ),
):
    # Initialize ByteRover service if memory features are enabled
    byterover_service = None
    memory_features_requested = memory_create or memory_retrieve
    if memory_features_requested:
        if not byterover_api_key or not byterover_user_id:
            print("Error: ByteRover API key and User ID are required when using memory features (--memory-create or --memory-retrieve)")
            return 1
        
        byterover_service = ByteroverService(byterover_api_key, byterover_user_id)
        print(f"ByteRover memory system initialized (create: {memory_create}, retrieve: {memory_retrieve})")

    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha[:7]
    if repo.is_dirty():
        commit_hash += "-dirty"

    if stats_only and not dirnames:
        latest_dir = find_latest_benchmark_dir()
        dirnames = [str(latest_dir)]

    if dirnames is None:
        dirnames = []

    if len(dirnames) > 1 and not (stats_only or diffs_only):
        print("Only provide 1 dirname unless running with --stats or --diffs")
        return 1

    updated_dirnames = []
    for dirname in dirnames:
        dirname = Path(dirname)
        dirname = resolve_dirname(dirname, stats_only or cont, make_new)
        if not dirname:
            return 1
        updated_dirnames.append(dirname)

    if stats_only:
        return show_stats(updated_dirnames, graphs, stats_languages)

    if diffs_only:
        return show_diffs(updated_dirnames)

    assert len(updated_dirnames) == 1, updated_dirnames
    dirname = updated_dirnames[0]

    if "AIDER_DOCKER" not in os.environ:
        print("Warning: benchmarking runs unvetted code from GPT, run in a docker container")
        return

    assert BENCHMARK_DNAME.exists() and BENCHMARK_DNAME.is_dir(), BENCHMARK_DNAME

    def get_exercise_dirs(base_dir, languages=None):
        """Get all exercise directories for specified languages (or all if none specified)"""
        base_dir = Path(base_dir)

        # Get available language dirs
        lang_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

        # Filter to requested languages if specified
        if languages:
            requested = set(lang.strip().lower() for lang in languages.split(","))
            lang_dirs = [d for d in lang_dirs if d.name.lower() in requested]
            dump(lang_dirs)
            if not lang_dirs:
                print(f"No matching language directories found for: {languages}")
                return []

        # Get all exercise dirs under exercises/practice for each language
        exercise_dirs = []
        for lang_dir in lang_dirs:
            practice_dir = lang_dir / "exercises" / "practice"
            if practice_dir.exists():
                exercise_dirs.extend(d for d in practice_dir.iterdir() if d.is_dir())

        return exercise_dirs

    original_dname = BENCHMARK_DNAME / exercises_dir
    assert original_dname.exists() and original_dname.is_dir(), original_dname

    exercise_dirs = get_exercise_dirs(original_dname, languages)

    if not exercise_dirs:
        print("No exercise directories found")
        return 1

    if clean and dirname.exists():
        print("Cleaning up and replacing", dirname)
        dir_files = set(fn.name for fn in dirname.glob("*"))
        original_files = set(fn.name for fn in original_dname.glob("*"))
        if dir_files != original_files:
            print("ERROR: will not delete dir that does not look like original tests", dirname)
            return

        dest = dirname.parent / "OLD" / dirname.name
        if dest.exists():
            old_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            dest = dirname.parent / "OLD" / (old_now + dirname.name)

        dirname.rename(dest)

    if not dirname.exists():
        print(f"Copying {original_dname} -> {dirname} ...")
        # Only copy the practice subdirs with exercises
        os.makedirs(dirname, exist_ok=True)
        for lang_dir in original_dname.iterdir():
            if not lang_dir.is_dir():
                continue
            practice_dir = lang_dir / "exercises" / "practice"
            if practice_dir.exists():
                dest_lang_dir = dirname / lang_dir.name / "exercises" / "practice"
                os.makedirs(dest_lang_dir.parent, exist_ok=True)
                shutil.copytree(practice_dir, dest_lang_dir)
        print("...done")

    test_dnames = sorted(str(d.relative_to(original_dname)) for d in exercise_dirs)

    resource_metadata = importlib_resources.files("aider.resources").joinpath("model-metadata.json")
    model_metadata_files_loaded = models.register_litellm_models([resource_metadata])
    dump(model_metadata_files_loaded)

    if read_model_settings:
        try:
            files_loaded = models.register_models([read_model_settings])
            if verbose:
                if files_loaded:
                    print(f"Loaded model settings from: {files_loaded[0]}")
                else:
                    print(f"No model settings loaded from: {read_model_settings}")
        except Exception as e:
            print(f"Error loading model settings: {e}")
            return 1

    if keywords:
        keywords = keywords.split(",")
        test_dnames = [dn for dn in test_dnames for keyword in keywords if keyword in dn]

    total_tests = len(test_dnames)
    
    # Set random seed if provided for reproducible test selection
    if seed is not None:
        random.seed(seed)
        
    random.shuffle(test_dnames)
    
    # Handle percentage-based test selection
    if num_tests > 0:
        # If num_tests is explicitly set, it takes precedence
        test_dnames = test_dnames[:num_tests]
        print(f"Running {len(test_dnames)} of {total_tests} tests (using --num-tests)")
    else:
        # Otherwise use percentage (default is now 5%)
        num_to_run = max(1, int(total_tests * percentage / 100.0))
        test_dnames = test_dnames[:num_to_run]
        print(f"Running {num_to_run} of {total_tests} tests ({percentage:.1f}%)")
        
        # If running 100%, make it clear
        if percentage == 100.0:
            print("Running ALL tests (100%)")

    # Don't give up when benchmarking
    LONG_TIMEOUT = 24 * 60 * 60
    sendchat.RETRY_TIMEOUT = LONG_TIMEOUT
    base_coder.RETRY_TIMEOUT = LONG_TIMEOUT
    models.RETRY_TIMEOUT = LONG_TIMEOUT

    if threads == 1:
        all_results = []
        for test_path in test_dnames:
            results = run_test(
                original_dname,
                dirname / test_path,
                model,
                edit_format,
                tries,
                no_unit_tests,
                no_aider,
                verbose,
                commit_hash,
                replay,
                editor_model,
                editor_edit_format,
                num_ctx,
                sleep,
                reasoning_effort,
                thinking_tokens,
                byterover_service=byterover_service,
                memory_create=memory_create,
                memory_retrieve=memory_retrieve,
                memory_limit=memory_limit,
            )

            all_results.append(results)
            summarize_results(dirname)
            if sleep:
                time.sleep(sleep)
    else:
        run_test_threaded = lox.thread(threads)(run_test)
        for test_path in test_dnames:
            run_test_threaded.scatter(
                original_dname,
                dirname / test_path,
                model,
                edit_format,
                tries,
                no_unit_tests,
                no_aider,
                verbose,
                commit_hash,
                replay,
                editor_model,
                editor_edit_format,
                num_ctx,
                sleep,
                reasoning_effort,
                thinking_tokens,
                byterover_service=byterover_service,
                memory_create=memory_create,
                memory_retrieve=memory_retrieve,
                memory_limit=memory_limit,
            )
        all_results = run_test_threaded.gather(tqdm=True)

    print()
    print()
    print()
    summarize_results(dirname)

    return 0


def show_diffs(dirnames):
    dirnames = sorted(dirnames)

    all_results = dict((dirname, load_results(dirname)) for dirname in dirnames)
    testcases = set()
    for results in all_results.values():
        testcases.update(result["testcase"] for result in results)

    testcases = sorted(testcases)

    unchanged = set()

    for testcase in testcases:
        all_outcomes = []
        for dirname in dirnames:
            results = all_results[dirname]
            result = [r for r in results if r["testcase"] == testcase][0]

            outcomes = tuple(result["tests_outcomes"])
            all_outcomes.append(True in outcomes)

        if len(set(all_outcomes)) == 1:
            unchanged.add(testcase)
            continue

        print()
        print(testcase)
        for outcome, dirname in zip(all_outcomes, dirnames):
            print(outcome, f"{dirname}/{testcase}/.aider.chat.history.md")

    changed = set(testcases) - unchanged
    print()
    print("changed:", len(changed), ",".join(sorted(changed)))
    print()
    print("unchanged:", len(unchanged), ",".join(sorted(unchanged)))


def load_results(dirname, stats_languages=None):
    dirname = Path(dirname)
    all_results = []

    if stats_languages:
        languages = [lang.strip().lower() for lang in stats_languages.split(",")]
        glob_patterns = [f"{lang}/exercises/practice/*/.aider.results.json" for lang in languages]
    else:
        glob_patterns = ["*/exercises/practice/*/.aider.results.json"]

    for pattern in glob_patterns:
        for fname in dirname.glob(pattern):
            try:
                results = json.loads(fname.read_text())
                all_results.append(results)
            except json.JSONDecodeError:
                print("json.JSONDecodeError", fname)
                continue
    return all_results


def summarize_results(dirname, stats_languages=None):
    all_results = load_results(dirname, stats_languages)

    res = SimpleNamespace()
    res.total_tests = len(list(Path(dirname).glob("*/exercises/practice/*")))

    try:
        tries = max(len(results.get("tests_outcomes", [])) for results in all_results if results)
    except ValueError:
        tries = 0

    res.dir_name = str(dirname)

    passed_tests = [0] * tries

    res.completed_tests = 0
    res.duration = 0
    res.cost = 0
    res.error_outputs = 0
    res.user_asks = 0
    res.test_timeouts = 0
    res.exhausted_context_windows = 0
    res.num_malformed_responses = 0
    res.num_with_malformed_responses = 0
    res.syntax_errors = 0
    res.indentation_errors = 0
    res.lazy_comments = 0
    res.prompt_tokens = 0
    res.completion_tokens = 0

    res.reasoning_effort = None
    res.thinking_tokens = None
    variants = defaultdict(set)

    for results in all_results:
        if not results:
            continue

        res.completed_tests += 1
        tests_outcomes = results.get("tests_outcomes", [])
        passed = tests_outcomes and tests_outcomes[-1]
        if passed:
            for i in range(len(tests_outcomes) - 1, tries):
                passed_tests[i] += 1

        res.cost += results.get("cost", 0)
        res.duration += results.get("duration", 0)
        res.test_timeouts += results.get("test_timeouts", 0)

        res.error_outputs += results.get("num_error_outputs", 0)
        res.user_asks += results.get("num_user_asks", 0)
        res.exhausted_context_windows += results.get("num_exhausted_context_windows", 0)
        res.num_malformed_responses += results.get("num_malformed_responses", 0)
        if results.get("num_malformed_responses"):
            res.num_with_malformed_responses += 1
        res.lazy_comments += results.get("lazy_comments", 0)

        res.syntax_errors += results.get("syntax_errors", 0)
        res.indentation_errors += results.get("indentation_errors", 0)

        res.prompt_tokens += results.get("prompt_tokens", 0)
        res.completion_tokens += results.get("completion_tokens", 0)

        res.reasoning_effort = results.get("reasoning_effort")
        res.thinking_tokens = results.get("thinking_tokens")

        for key in "model edit_format commit_hash editor_model editor_edit_format".split():
            val = results.get(key)
            if val:
                variants[key].add(val)

    if not res.completed_tests:
        return

    # if res.completed_tests < 133:
    #    return

    console = Console(highlight=False)
    console.rule(title=str(dirname))

    commit_hashes = variants["commit_hash"]
    versions = get_versions(commit_hashes)
    date = dirname.name[:10]

    def show(stat, red="red"):
        val = getattr(res, stat)
        style = red if val else None
        console.print(f"  {stat}: {val}", style=style)

    percents = dict()
    for i in range(tries):
        pass_rate = 100 * passed_tests[i] / res.completed_tests
        percents[i] = pass_rate
        # console.print(f"{pass_rate:.1f}% correct after try {i+1}")
        setattr(res, f"pass_rate_{i + 1}", f"{pass_rate:.1f}")
        setattr(res, f"pass_num_{i + 1}", passed_tests[i])

    print(f"- dirname: {dirname.name}")
    style = None if res.completed_tests == res.total_tests else "red"
    console.print(f"  test_cases: {res.completed_tests}", style=style)
    for key, val in variants.items():
        if len(val) > 1:
            style = "red"
        else:
            style = None
        val = ", ".join(map(str, val))
        setattr(res, key, val)
        console.print(f"  {key}: {val}", style=style)

    if res.reasoning_effort is not None:
        print(f"  reasoning_effort: {res.reasoning_effort}")
    if res.thinking_tokens is not None:
        print(f"  thinking_tokens: {res.thinking_tokens}")

    for i in range(tries):
        print(f"  pass_rate_{i + 1}: {percents[i]:.1f}")
    for i in range(tries):
        print(f"  pass_num_{i + 1}: {passed_tests[i]}")

    pct_well_formed = 1.0 - res.num_with_malformed_responses / res.completed_tests
    print(f"  percent_cases_well_formed: {pct_well_formed * 100:.1f}")

    show("error_outputs")
    show("num_malformed_responses")
    show("num_with_malformed_responses")
    show("user_asks")
    show("lazy_comments")
    show("syntax_errors")
    show("indentation_errors")
    show("exhausted_context_windows")
    show("prompt_tokens", red=None)
    show("completion_tokens", red=None)
    show("test_timeouts")
    print(f"  total_tests: {res.total_tests}")

    if variants["model"]:
        a_model = set(variants["model"]).pop()
        command = f"aider --model {a_model}"
        print(f"  command: {command}")

    print(f"  date: {date}")
    print("  versions:", ",".join(versions))

    res.avg_duration = res.duration / res.completed_tests
    print(f"  seconds_per_case: {res.avg_duration:.1f}")

    print(f"  total_cost: {res.cost:.4f}")

    res.avg_cost = res.cost / res.completed_tests

    projected_cost = res.avg_cost * res.total_tests

    print()
    print(
        f"costs: ${res.avg_cost:.4f}/test-case, ${res.cost:.2f} total,"
        f" ${projected_cost:.2f} projected"
    )

    console.rule()

    # print(json.dumps(vars(res), indent=4, sort_keys=True))
    return res


def get_versions(commit_hashes):
    versions = set()
    for hsh in commit_hashes:
        if not hsh:
            continue
        hsh = hsh.split("-")[0]
        try:
            version = subprocess.check_output(
                ["git", "show", f"{hsh}:aider/__init__.py"], universal_newlines=True
            )
            version = re.search(r'__version__ = "(.*)"', version).group(1)
            versions.add(version)
        except subprocess.CalledProcessError:
            pass
    return versions


def get_replayed_content(replay_dname, test_dname):
    replay_dname = Path(replay_dname)
    test_dname = Path(test_dname)
    dump(replay_dname, test_dname)

    test_name = test_dname.name
    replay_fname = replay_dname / test_name / ".aider.chat.history.md"
    dump(replay_fname)

    res = replay_fname.read_text()
    return res

    res = res.splitlines(keepends=True)
    res = [line for line in res if not line.startswith("> ") and not line.startswith("#### ")]
    return "".join(res)


def run_test(original_dname, testdir, *args, **kwargs):
    try:
        return run_test_real(original_dname, testdir, *args, **kwargs)
    except Exception:
        print("=" * 40)
        print("Test failed")
        traceback.print_exc()

        testdir = Path(testdir)
        results_fname = testdir / ".aider.results.json"
        results_fname.write_text(json.dumps(dict(exception=traceback.format_exc())))


def run_test_real(
    original_dname,
    testdir,
    model_name,
    edit_format,
    tries,
    no_unit_tests,
    no_aider,
    verbose,
    commit_hash,
    replay,
    editor_model,
    editor_edit_format,
    num_ctx=None,
    sleep=0,
    reasoning_effort: Optional[str] = None,
    thinking_tokens: Optional[int] = None,
    read_model_settings=None,
    byterover_service=None,
    memory_create=False,
    memory_retrieve=False,
    memory_limit=3,
):
    if not os.path.isdir(testdir):
        print("Not a dir:", testdir)
        return

    testdir = Path(testdir)

    history_fname = testdir / ".aider.chat.history.md"

    results_fname = testdir / ".aider.results.json"
    if results_fname.exists():
        try:
            res = json.loads(results_fname.read_text())
            # if res.get("test_timeouts", 0) > 0:
            #    print(f"{results_fname} test timeouts, redoing...")
            # else:
            return res
        except JSONDecodeError:
            print(f"{results_fname} failed to parse, redoing...")

    # Read solution and test files from config
    fnames = []
    config_file = testdir / ".meta/config.json"
    if not config_file.exists():
        raise ValueError(f"No config file found: {config_file}")

    with open(config_file) as f:
        config = json.loads(f.read())

    # Get file sets from config
    test_files = config.get("files", {}).get("test", [])
    example_files = config.get("files", {}).get("example", [])
    solution_files = set(config.get("files", {}).get("solution", []))

    # Forcibly ignore certain files not covered by test_files and example_files
    ignore_files = set(
        [
            "CMakeLists.txt",
            "Cargo.toml",
        ]
    )

    # Add all files under .meta and .docs directories
    ignore_files.update(str(p.relative_to(testdir)) for p in testdir.glob(".meta/**/*"))
    ignore_files.update(str(p.relative_to(testdir)) for p in testdir.glob(".docs/**/*"))

    # Also ignore test & example files
    ignore_files.update(test_files)
    ignore_files.update(example_files)

    # Remove any ignore files from the solution set that LLM will edit
    solution_files.difference_update(ignore_files)

    # Copy all solution files
    for file_path in solution_files:
        src = testdir / Path(file_path)
        if src.exists():
            fnames.append(src)
            # restore the original file, in case we interrupted a prev run
            # Find the original file in the language-specific practice dir
            lang_part = str(testdir).split("/exercises/practice/")[0]
            original_fname = (
                original_dname
                / Path(lang_part).name
                / "exercises"
                / "practice"
                / testdir.name
                / file_path
            )
            if original_fname.exists():
                os.makedirs(src.parent, exist_ok=True)
                shutil.copy(original_fname, src)
        else:
            print(f"Warning: Solution file not found: {src}")

    file_list = " ".join(fname.name for fname in fnames)

    instructions = ""

    introduction = testdir / ".docs/introduction.md"
    if introduction.exists():
        instructions += introduction.read_text()
    instructions += (testdir / ".docs/instructions.md").read_text()
    instructions_append = testdir / ".docs/instructions.append.md"
    if instructions_append.exists():
        instructions += instructions_append.read_text()

    instructions += prompts.instructions_addendum.format(file_list=file_list)
    
    # Retrieve relevant memories if enabled
    memory_results = {"results": []}
    if memory_retrieve and byterover_service:
        try:
            # Create a search query based on the exercise
            search_query = f"Exercise: {testdir.name} {instructions[:200]}"
            
            # Retrieve relevant memories
            memory_results = byterover_service.search_memories(search_query, memory_limit)
            if memory_results and memory_results.get("results"):
                # Format memories to append to instructions
                memory_content = "\n\nRelevant past experiences:\n"
                for idx, result in enumerate(memory_results["results"]):
                    memory_content += f"\n{idx + 1}. {result['memory']}\n"
                
                # Append memories to instructions
                instructions += memory_content
                print(f"Added {len(memory_results['results'])} memories to instructions for {testdir.name}")
        except Exception as e:
            print(f"Error retrieving memories: {e}")

    io = InputOutput(
        pretty=False,
        yes=True,
        chat_history_file=history_fname,
    )

    # weak_model_name = model_name
    weak_model_name = None

    main_model = models.Model(
        model_name,
        weak_model=weak_model_name,
        editor_model=editor_model,
        editor_edit_format=editor_edit_format,
        verbose=verbose,
    )

    if reasoning_effort is not None:
        main_model.set_reasoning_effort(reasoning_effort)

    if thinking_tokens is not None:
        main_model.set_thinking_tokens(thinking_tokens)

    dump(main_model.max_chat_history_tokens)

    if num_ctx:
        if not main_model.extra_params:
            main_model.extra_params = {}
        main_model.extra_params["num_ctx"] = num_ctx
    edit_format = edit_format or main_model.edit_format

    dump(main_model)
    dump(edit_format)
    show_fnames = ",".join(map(str, fnames))
    print("fnames:", show_fnames)

    coder = Coder.create(
        main_model,
        edit_format,
        io,
        fnames=fnames,
        use_git=False,
        stream=False,
        verbose=verbose,
        # auto_lint=False,  # disabled for code-in-json experiments
        cache_prompts=True,
        suggest_shell_commands=False,
        ignore_mentions=ignore_files,
    )
    dump(coder.ignore_mentions)

    coder.show_announcements()
    coder.get_file_mentions = lambda x: set()  # No loading of any other files

    timeouts = 0

    syntax_errors = 0
    indentation_errors = 0
    lazy_comments = 0

    dur = 0
    test_outcomes = []
    for i in range(tries):
        start = time.time()

        if no_aider:
            pass
        elif replay:
            response = get_replayed_content(replay, testdir)
            coder.partial_response_content = response

            show = response.splitlines(keepends=True)
            show = [">> " + line for line in show]
            io.append_chat_history("".join(show))

            coder.apply_updates()
        else:
            response = coder.run(with_message=instructions, preproc=False)

        dur += time.time() - start

        if not no_aider:
            pat = r"^[+]? *[#].* [.][.][.] "
            # Count the number of lines that match pat in response
            dump(response)
            lazy_comments += len(re.findall(pat, response, re.MULTILINE))
            dump(lazy_comments)

        if coder.last_keyboard_interrupt:
            raise KeyboardInterrupt

        if no_unit_tests:
            break

        try:
            errors = run_unit_tests(original_dname, testdir, history_fname, test_files)
        except subprocess.TimeoutExpired:
            # try:
            #    errors = run_unit_tests(original_dname, testdir, history_fname, test_files)
            # except subprocess.TimeoutExpired:
            errors = "Tests timed out!"
            timeouts += 1

        if errors:
            test_outcomes.append(False)
        else:
            test_outcomes.append(True)
            break

        if replay:
            io.append_chat_history(errors)

        errors = errors.splitlines()

        syntax_errors += sum(1 for line in errors if line.startswith("SyntaxError"))
        indentation_errors += sum(1 for line in errors if line.startswith("IndentationError"))

        print(errors[-1])
        errors = "\n".join(errors)
        instructions = errors
        instructions += prompts.test_failures.format(file_list=file_list)

    # Clean up build directories after all attempts
    # Rust target/debug
    target_dir = testdir / "target" / "debug"
    if target_dir.exists():
        try:
            shutil.rmtree(target_dir)
            if verbose:
                print(f"Cleaned up Rust target/debug directory: {target_dir}")
        except (OSError, shutil.Error, PermissionError) as e:
            if verbose:
                print(f"Failed to clean up Rust target/debug directory: {e}")

    # Java build directories
    java_build_dir = testdir / "build"
    if java_build_dir.exists():
        try:
            shutil.rmtree(java_build_dir)
            if verbose:
                print(f"Cleaned up Java build directory: {java_build_dir}")
        except (OSError, shutil.Error, PermissionError) as e:
            if verbose:
                print(f"Failed to clean up Java build directory: {e}")

    # Node.js node_modules directories
    node_modules_dir = testdir / "node_modules"
    if node_modules_dir.exists():
        try:
            shutil.rmtree(node_modules_dir)
            if verbose:
                print(f"Cleaned up Node.js node_modules directory: {node_modules_dir}")
        except (OSError, shutil.Error, PermissionError) as e:
            if verbose:
                print(f"Failed to clean up Node.js node_modules directory: {e}")

    # Create memory from the exercise and outcome if enabled
    if memory_create and byterover_service and not no_aider:
        try:
            # Create memory from the exercise and outcome
            outcome_str = "successful" if True in test_outcomes else "unsuccessful"
            memory_messages = [
                {"role": "user", "content": instructions[:1000]},  # Truncate if too long
                {"role": "assistant", "content": response[:1000] if 'response' in locals() else "No response generated"}  # Truncate if too long
            ]
            
            # Add test outcome information
            summary = f"Exercise '{testdir.name}' was {outcome_str}. "
            if True in test_outcomes:
                summary += f"Passed on attempt {test_outcomes.index(True) + 1} of {tries}."
            else:
                summary += f"Failed after {tries} attempts."
            
            # Create the memory
            byterover_service.create_memory(memory_messages)
            print(f"Created memory for {testdir.name}")
        except Exception as e:
            print(f"Error creating memory: {e}")

    results = dict(
        testdir=str(testdir),
        testcase=testdir.name,
        model=main_model.name,
        edit_format=edit_format,
        tests_outcomes=test_outcomes,
        cost=coder.total_cost,
        duration=dur,
        test_timeouts=timeouts,
        commit_hash=commit_hash,
        num_error_outputs=io.num_error_outputs,
        num_user_asks=io.num_user_asks,
        num_exhausted_context_windows=coder.num_exhausted_context_windows,
        num_malformed_responses=coder.num_malformed_responses,
        syntax_errors=syntax_errors,
        indentation_errors=indentation_errors,
        lazy_comments=lazy_comments,  # Add the count of pattern matches to the results
        reasoning_effort=reasoning_effort,
        prompt_tokens=coder.total_tokens_sent,
        completion_tokens=coder.total_tokens_received,
        thinking_tokens=thinking_tokens,
        chat_hashes=list(
            zip(
                coder.chat_completion_call_hashes,
                coder.chat_completion_response_hashes,
            )
        ),
    )
    
    # Add memory-related metrics to results
    if byterover_service:
        results.update({
            "memory_create_enabled": memory_create,
            "memory_retrieve_enabled": memory_retrieve,
            "memories_used": len(memory_results.get("results", [])) if memory_retrieve else 0,
        })

    if edit_format == "architect":
        results["editor_model"] = main_model.editor_model.name if main_model.editor_model else None
        results["editor_edit_format"] = main_model.editor_edit_format
    dump(results)

    results_fname.write_text(json.dumps(results, indent=4))

    return results


def run_unit_tests(original_dname, testdir, history_fname, test_files):
    timeout = 60 * 3

    # Map of file extensions to test commands
    TEST_COMMANDS = {
        ".py": ["pytest"],
        ".rs": ["cargo", "test", "--", "--include-ignored"],
        ".go": ["go", "test", "./..."],
        ".js": ["/aider/benchmark/npm-test.sh"],
        ".cpp": ["/aider/benchmark/cpp-test.sh"],
        ".java": ["./gradlew", "test"],
    }

    # Get unique file extensions from test files
    extensions = {Path(f).suffix for f in test_files}

    # Find matching test command
    command = None
    for ext in extensions:
        if ext in TEST_COMMANDS:
            command = TEST_COMMANDS[ext]
            break

    if not command:
        raise ValueError(f"No test command found for files with extensions: {extensions}")

    # Copy test files from original directory
    for file_path in test_files:
        src = original_dname / Path(*testdir.parts[-4:]) / file_path
        dst = testdir / file_path
        if src.exists():
            print("copying", src, dst)
            os.makedirs(dst.parent, exist_ok=True)
            shutil.copy(src, dst)

    # Remove @Disabled annotations from Java test files
    for file_path in test_files:
        if file_path.endswith(".java"):
            test_file = testdir / file_path
            if test_file.exists():
                content = test_file.read_text()
                content = re.sub(r"@Disabled\([^)]*\)\s*\n", "", content)
                test_file.write_text(content)

    print(" ".join(command))

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        cwd=testdir,
        encoding="utf-8",
        errors="replace",
    )

    success = result.returncode == 0
    res = result.stdout
    res = cleanup_test_output(res, testdir)
    dump(res)

    with history_fname.open("a") as fh:
        fh.write(f"```\n{res}\n```")

    if not success:
        print(f"Tests failed: {testdir}")
        return res


def cleanup_test_output(output, testdir):
    # remove timing info, to avoid randomizing the response to GPT
    res = re.sub(r"\bin \d+\.\d+s\b", "", output)
    res = res.replace(str(testdir), str(testdir.name))
    return res


@app.command()
def test_memory(
    # Exercise selection
    exercises_dir: str = typer.Option(EXERCISES_DIR_DEFAULT, "--exercises-dir", help="Directory with exercise files"),
    languages: str = typer.Option(None, "--languages", "-l", help="Only test specific languages (comma separated)"),
    keywords: str = typer.Option(None, "--keywords", "-k", help="Only test exercises containing keywords (comma sep)"),
    
    # Test selection 
    num_tests: int = typer.Option(-1, "--num-tests", help="Number of exercises to test"),
    percentage: float = typer.Option(100.0, "--percentage", "-p", help="Percentage of exercises to test (1-100)", min=1.0, max=100.0),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducible selection"),
    
    # Memory system
    byterover_api_key: str = typer.Option(None, "--byterover-api-key", help="ByteRover API key", envvar="BYTEROVER_API_KEY"),
    byterover_user_id: str = typer.Option(None, "--byterover-user-id", help="ByteRover User ID", envvar="BYTEROVER_USER_ID"),
    memory_limit: int = typer.Option(3, "--memory-limit", help="Maximum number of memories to retrieve"),
    
    # Output options
    output_csv: str = typer.Option("memory_test_results.csv", "--output-csv", "-o", help="Path to output CSV file"),
    append_mode: bool = typer.Option(False, "--append", help="Append to existing CSV instead of overwriting"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Test memory retrieval for exercises and output results to CSV"""
    
    # Validate ByteRover credentials
    if not byterover_api_key or not byterover_user_id:
        print("Error: ByteRover API key and User ID are required for memory testing")
        print("Use --byterover-api-key and --byterover-user-id or set BYTEROVER_API_KEY and BYTEROVER_USER_ID environment variables")
        return 1
    
    # Initialize ByteRover service
    byterover_service = ByteroverService(byterover_api_key, byterover_user_id)
    if not byterover_service.is_service_configured():
        print("Error: Failed to configure ByteRover service")
        return 1
    
    print(f"ByteRover memory testing initialized (limit: {memory_limit})")
    
    # Run the memory retrieval test
    try:
        results = test_memory_retrieval(
            exercises_dir=exercises_dir,
            languages=languages,
            keywords=keywords,
            num_tests=num_tests,
            percentage=percentage,
            seed=seed,
            memory_limit=memory_limit,
            byterover_service=byterover_service,
            verbose=verbose,
        )
        
        # Write results to CSV
        write_results_to_csv(results, output_csv, append_mode)
        
        # Print summary statistics
        print_memory_test_summary(results)
        
        return 0
        
    except Exception as e:
        print(f"Error during memory testing: {e}")
        traceback.print_exc()
        return 1


def test_memory_retrieval(
    exercises_dir: str,
    languages: Optional[str] = None,
    keywords: Optional[str] = None,
    num_tests: int = -1,
    percentage: float = 100.0,
    seed: Optional[int] = None,
    memory_limit: int = 3,
    byterover_service: ByteroverService = None,
    verbose: bool = False,
) -> List[dict]:
    """Test memory retrieval for selected exercises and return results"""
    
    original_dname = BENCHMARK_DNAME / exercises_dir
    if not original_dname.exists():
        raise ValueError(f"Exercises directory not found: {original_dname}")
    
    # Get exercise directories using existing logic
    def get_exercise_dirs(base_dir, languages=None):
        """Get all exercise directories for specified languages (or all if none specified)"""
        base_dir = Path(base_dir)

        # Get available language dirs
        lang_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

        # Filter to requested languages if specified
        if languages:
            requested = set(lang.strip().lower() for lang in languages.split(","))
            lang_dirs = [d for d in lang_dirs if d.name.lower() in requested]
            if not lang_dirs:
                print(f"No matching language directories found for: {languages}")
                return []

        # Get all exercise dirs under exercises/practice for each language
        exercise_dirs = []
        for lang_dir in lang_dirs:
            practice_dir = lang_dir / "exercises" / "practice"
            if practice_dir.exists():
                exercise_dirs.extend(d for d in practice_dir.iterdir() if d.is_dir())

        return exercise_dirs
    
    exercise_dirs = get_exercise_dirs(original_dname, languages)
    
    if not exercise_dirs:
        print("No exercise directories found")
        return []
    
    # Apply keyword filtering
    test_dnames = sorted(str(d.relative_to(original_dname)) for d in exercise_dirs)
    
    if keywords:
        keywords_list = keywords.split(",")
        test_dnames = [dn for dn in test_dnames for keyword in keywords_list if keyword in dn]
    
    total_tests = len(test_dnames)
    
    # Set random seed if provided for reproducible test selection
    if seed is not None:
        random.seed(seed)
        
    random.shuffle(test_dnames)
    
    # Handle test selection
    if num_tests > 0:
        # If num_tests is explicitly set, it takes precedence
        test_dnames = test_dnames[:num_tests]
        print(f"Testing {len(test_dnames)} of {total_tests} exercises (using --num-tests)")
    else:
        # Otherwise use percentage
        num_to_run = max(1, int(total_tests * percentage / 100.0))
        test_dnames = test_dnames[:num_to_run]
        print(f"Testing {num_to_run} of {total_tests} exercises ({percentage:.1f}%)")
    
    # Process each exercise
    results = []
    console = Console(highlight=False) if verbose else None
    
    for i, test_path in enumerate(test_dnames):
        if verbose and console:
            console.print(f"[{i + 1}/{len(test_dnames)}] Processing {test_path}")
        
        try:
            testdir = original_dname / test_path
            result = test_single_exercise_memory(testdir, test_path, memory_limit, byterover_service, verbose)
            results.append(result)
            
        except Exception as e:
            if verbose:
                print(f"Error processing {test_path}: {e}")
            # Still record the error in results
            results.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'exercise_name': Path(test_path).name,
                'language': test_path.split('/')[0] if '/' in test_path else 'unknown',
                'exercise_path': test_path,
                'instruction_preview': '',
                'full_instruction_length': 0,
                'search_query': '',
                'num_memories_found': 0,
                'api_error': str(e),
                **{f'memory_{j + 1}_content': '' for j in range(memory_limit)},
                **{f'memory_{j + 1}_score': 0.0 for j in range(memory_limit)},
                **{f'memory_{j + 1}_id': '' for j in range(memory_limit)},
            })
    
    return results


def test_single_exercise_memory(testdir: Path, test_path: str, memory_limit: int, byterover_service: ByteroverService, verbose: bool = False) -> dict:
    """Test memory retrieval for a single exercise"""
    
    # Load exercise instructions using same logic as run_test_real
    instructions = load_exercise_instructions(testdir)
    
    # Create search query using same logic as main benchmark
    search_query = create_search_query(testdir.name, instructions)
    
    # Initialize result structure
    result = {
        'timestamp': datetime.datetime.now().isoformat(),
        'exercise_name': testdir.name,
        'language': test_path.split('/')[0] if '/' in test_path else 'unknown',
        'exercise_path': test_path,
        'instruction_preview': instructions[:200] if instructions else '',
        'full_instruction_length': len(instructions),
        'search_query': search_query,
        'num_memories_found': 0,
        'api_error': '',
    }
    
    # Initialize memory fields
    for j in range(memory_limit):
        result[f'memory_{j + 1}_content'] = ''
        result[f'memory_{j + 1}_score'] = 0.0
        result[f'memory_{j + 1}_id'] = ''
    
    # Retrieve memories
    try:
        memory_results = byterover_service.search_memories(search_query, memory_limit)
        
        if memory_results and memory_results.get("results"):
            memories = memory_results["results"]
            result['num_memories_found'] = len(memories)
            
            # Store individual memory details
            for j, memory in enumerate(memories[:memory_limit]):
                result[f'memory_{j + 1}_content'] = memory.get('memory', '')[:500]  # Truncate for CSV
                result[f'memory_{j + 1}_score'] = memory.get('score', 0.0)
                result[f'memory_{j + 1}_id'] = memory.get('id', '')
            
            if verbose:
                print(f"  Found {len(memories)} memories for {testdir.name}")
        else:
            if verbose:
                print(f"  No memories found for {testdir.name}")
                
    except Exception as e:
        result['api_error'] = str(e)
        if verbose:
            print(f"  API error for {testdir.name}: {e}")
    
    return result


def load_exercise_instructions(testdir: Path) -> str:
    """Load exercise instructions from .docs files"""
    instructions = ""
    
    try:
        introduction = testdir / ".docs/introduction.md"
        if introduction.exists():
            instructions += introduction.read_text()
        
        instructions_file = testdir / ".docs/instructions.md"
        if instructions_file.exists():
            instructions += instructions_file.read_text()
        
        instructions_append = testdir / ".docs/instructions.append.md"
        if instructions_append.exists():
            instructions += instructions_append.read_text()
            
    except Exception as e:
        print(f"Warning: Error loading instructions for {testdir}: {e}")
    
    return instructions


def create_search_query(exercise_name: str, instructions: str) -> str:
    """Create search query using same logic as main benchmark"""
    return f"Exercise: {exercise_name} {instructions[:200]}"


def write_results_to_csv(results: List[dict], output_path: str, append_mode: bool = False):
    """Write memory test results to CSV file"""
    if not results:
        print("No results to write")
        return
    
    import csv
    
    # Define field order for CSV
    base_fields = [
        'timestamp', 'exercise_name', 'language', 'exercise_path', 
        'instruction_preview', 'full_instruction_length', 'search_query',
        'num_memories_found', 'api_error'
    ]
    
    # Add memory fields based on the first result
    memory_limit = 0
    for key in results[0].keys():
        if key.startswith('memory_') and key.endswith('_content'):
            memory_limit += 1
    
    memory_fields = []
    for j in range(memory_limit):
        memory_fields.extend([
            f'memory_{j + 1}_content',
            f'memory_{j + 1}_score', 
            f'memory_{j + 1}_id'
        ])
    
    fieldnames = base_fields + memory_fields
    
    # Determine write mode
    file_exists = Path(output_path).exists()
    write_mode = 'a' if (append_mode and file_exists) else 'w'
    write_header = not (append_mode and file_exists)
    
    try:
        with open(output_path, write_mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
            
            for result in results:
                # Ensure all fields are present
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"Results written to {output_path} ({len(results)} rows)")
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        raise


def print_memory_test_summary(results: List[dict]):
    """Print summary statistics for memory test results"""
    if not results:
        return
    
    total_exercises = len(results)
    exercises_with_memories = sum(1 for r in results if r['num_memories_found'] > 0)
    exercises_with_errors = sum(1 for r in results if r['api_error'])
    total_memories = sum(r['num_memories_found'] for r in results)
    
    print("\n" + "="*50)
    print("MEMORY RETRIEVAL TEST SUMMARY")
    print("="*50)
    print(f"Total exercises tested: {total_exercises}")
    print(f"Exercises with memories found: {exercises_with_memories} ({exercises_with_memories / total_exercises * 100:.1f}%)")
    print(f"Exercises with no memories: {total_exercises - exercises_with_memories} ({(total_exercises - exercises_with_memories) / total_exercises * 100:.1f}%)")
    print(f"Exercises with API errors: {exercises_with_errors} ({exercises_with_errors / total_exercises * 100:.1f}%)")
    print(f"Total memories retrieved: {total_memories}")
    print(f"Average memories per exercise: {total_memories / total_exercises:.2f}")
    
    if exercises_with_memories > 0:
        avg_memories_when_found = sum(r['num_memories_found'] for r in results if r['num_memories_found'] > 0) / exercises_with_memories
        print(f"Average memories when found: {avg_memories_when_found:.2f}")
    
    # Language breakdown
    language_stats = {}
    for result in results:
        lang = result['language']
        if lang not in language_stats:
            language_stats[lang] = {'total': 0, 'with_memories': 0}
        language_stats[lang]['total'] += 1
        if result['num_memories_found'] > 0:
            language_stats[lang]['with_memories'] += 1
    
    if len(language_stats) > 1:
        print(f"\nLanguage breakdown:")
        for lang, stats in sorted(language_stats.items()):
            pct = stats['with_memories'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {lang}: {stats['with_memories']}/{stats['total']} ({pct:.1f}%)")
    
    print("="*50)


if __name__ == "__main__":
    app()
