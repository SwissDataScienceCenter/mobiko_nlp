from enum import Enum

class EntityLabel(Enum):
    """Supported entity label types."""
    TAXON = "TAXON"
    HABITAT = "HABITAT"
    ENV_FEATURE = "ENV_FEATURE"
    THREAT = "THREAT"
    POPULATION = "POPULATION"
    LOCATION = "LOCATION"
    # BEHAVIOR = "BEHAVIOR"
    # DRIVER = "DRIVER"
    # STATUS = "STATUS"
    # CONTEXT = "CONTEXT"


def build_bio_labels():
    labels = ["O"]
    for e in EntityLabel:
        labels.append(f"B-{e.value}")
        labels.append(f"I-{e.value}")
    return labels


BIO_LABELS = build_bio_labels()