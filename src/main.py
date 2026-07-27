import json
from src.models import supply_chain
from src.models import skip_tracing
from src.models import segmentation
from src.models import identity_resolution


def main():
    report = dict(
        supply_chain=supply_chain.run(),
        skip_tracing=skip_tracing.run(),
        segmentation=segmentation.run(),
        identity_resolution=identity_resolution.run(),
    )
    with open("output/metrics_all.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved metrics_all.json")


if __name__ == "__main__":
    main()
