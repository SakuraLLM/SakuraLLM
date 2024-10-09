from typing import List
from packaging.version import Version


def is_version_compatible(target_version: str, versions: List[str]) -> bool:
    target = Version(target_version)

    return any(
        (target.major == v.major and target.minor == v.minor)
        for v in map(Version, versions)
    )
