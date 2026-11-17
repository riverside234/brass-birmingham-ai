# Legacy scripts

These scripts are available for historical reproduction or reference.
Sealed lab bundle manifests retain their original paths. Use the checkout
matching the bundle when reproducing its qualification.

| Script | Original workflow |
| --- | --- |
| [register-legacy-candidates.py](register-legacy-candidates.py) | R3 candidate registration and opponent banks. |
| [smoke-searchlab.py](smoke-searchlab.py) | Retired SearchLab smoke command. Prints the migration message. |
| [verify-searchlab.py](verify-searchlab.py) | Retired SearchLab verification command. Prints the migration message. |

The [R3 pilot summarizer](../reports/summarize-legacy.py) lives with the other
reports. The [retired R4/R5 launcher](../experiments/searchlab/r4r5.py) is indexed
under experiments.

The [Rust lab archive](rust-lab/README.md) groups 40 retired bundle,
qualification, and profiling tools with their shared legacy helpers.
