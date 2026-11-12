"""Where raw files, intermediate tables and results live. Override with CRASH_RAW, CRASH_INTERIM, CRASH_RESULTS."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("CRASH_RAW", os.path.join(ROOT, "data", "raw"))
INTERIM = os.environ.get("CRASH_INTERIM", os.path.join(ROOT, "data", "interim"))
RESULTS = os.environ.get("CRASH_RESULTS", os.path.join(ROOT, "results"))
MODELS = os.path.join(INTERIM, "models")


def result(name):
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def interim(name):
    os.makedirs(INTERIM, exist_ok=True)
    return os.path.join(INTERIM, name)


def model(name):
    os.makedirs(MODELS, exist_ok=True)
    return os.path.join(MODELS, name)
