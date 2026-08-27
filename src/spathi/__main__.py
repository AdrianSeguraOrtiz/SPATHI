"""Allow ``python -m spathi`` to behave like the console command."""

from spathi.cli import main

raise SystemExit(main())
