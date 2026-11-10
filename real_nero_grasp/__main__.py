"""Allow ``python -m real_nero_grasp`` to use the preserved CLI."""

from .runtime import main


if __name__ == "__main__":
    raise SystemExit(main())
