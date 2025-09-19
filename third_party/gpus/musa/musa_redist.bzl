load(
    "@xla//third_party/gpus/musa:musa_redist_ubuntu_20_04.bzl",
    "musa_redist_ubuntu_20_04",
)
load(
    "@xla//third_party/gpus/musa:musa_redist_ubuntu_22_04.bzl",
    "musa_redist_ubuntu_22_04",
)
load(
    "@xla//third_party/gpus/musa:musa_redist_ubuntu_24_04.bzl",
    "musa_redist_ubuntu_24_04",
)

musa_redist = {
    "ubuntu_20.04": musa_redist_ubuntu_20_04,
    "ubuntu_22.04": musa_redist_ubuntu_22_04,
    "ubuntu_24.04": musa_redist_ubuntu_24_04,
}
