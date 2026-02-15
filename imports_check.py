"""Quick import smoke test for the NLP learning environment."""

import importlib
import time
import collections
import pickle

def main() -> None:
    static_modules = {
        "importlib": importlib,
        "time": time,
        "collections": collections,
        "pickle": pickle,
    }
    dynamic_imports = [
        "tomotopy",
        "gensim",
        "gensim.models",
        "matplotlib.pyplot",
        "matplotlib.colors",
        "wordcloud",
    ]

    print("Always-available modules:")
    for name in static_modules:
        print(f"- {name}")

    print("\nThird-party module check:")
    missing = []
    for mod in dynamic_imports:
        try:
            importlib.import_module(mod)
            print(f"- OK: {mod}")
        except Exception as exc:
            missing.append((mod, str(exc)))
            print(f"- MISSING: {mod}")

    if missing:
        print("\nMissing or broken modules:")
        for mod, err in missing:
            print(f"- {mod}: {err}")
    else:
        print("\nAll third-party imports succeeded.")


if __name__ == "__main__":
    main()
