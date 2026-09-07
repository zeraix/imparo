from __future__ import annotations

import struct

from .errors import BuildError


def section_names(cubin: bytes) -> list[str]:
    if len(cubin) < 64 or cubin[:4] != b"\x7fELF":
        raise BuildError("module is not an ELF cubin")
    if cubin[4] != 2 or cubin[5] != 1:
        raise BuildError("module must be little-endian ELF64")
    try:
        section_offset = struct.unpack_from("<Q", cubin, 40)[0]
        section_size = struct.unpack_from("<H", cubin, 58)[0]
        section_count = struct.unpack_from("<H", cubin, 60)[0]
        string_index = struct.unpack_from("<H", cubin, 62)[0]
    except struct.error as error:
        raise BuildError("truncated ELF header") from error
    if section_size < 64 or not section_count or string_index >= section_count:
        raise BuildError("invalid ELF section table")
    table_end = section_offset + section_size * section_count
    if table_end > len(cubin):
        raise BuildError("truncated ELF section table")
    string_header = section_offset + section_size * string_index
    string_offset = struct.unpack_from("<Q", cubin, string_header + 24)[0]
    string_size = struct.unpack_from("<Q", cubin, string_header + 32)[0]
    if string_offset + string_size > len(cubin):
        raise BuildError("truncated ELF section-name table")
    strings = cubin[string_offset : string_offset + string_size]
    names: list[str] = []
    for index in range(section_count):
        header = section_offset + section_size * index
        name_offset = struct.unpack_from("<I", cubin, header)[0]
        if name_offset >= len(strings):
            raise BuildError("invalid ELF section name")
        end = strings.find(b"\0", name_offset)
        if end < 0:
            raise BuildError("unterminated ELF section name")
        try:
            names.append(strings[name_offset:end].decode("ascii"))
        except UnicodeDecodeError as error:
            raise BuildError("non-ASCII ELF section name") from error
    return names


def verify_release_cubin(
    cubin: bytes,
    expected_sm: int,
    forbidden_source_path: bytes | None = None,
) -> None:
    # CUDA cubins are ELF64 objects with NVIDIA's OSABI and EM_CUDA machine.
    # The low byte of e_flags is the real-SM code version.  Verify this from
    # the binary rather than trusting the requested compiler target.
    if cubin[7] != 51:
        raise BuildError("module does not use NVIDIA CUDA ELF OSABI")
    machine = struct.unpack_from("<H", cubin, 18)[0]
    if machine != 190:
        raise BuildError("module ELF machine is not EM_CUDA")
    flags = struct.unpack_from("<I", cubin, 48)[0]
    actual_sm = flags & 0xFF
    if actual_sm != expected_sm:
        raise BuildError(
            f"module real-SM {actual_sm} differs from expected SM{expected_sm}"
        )
    forbidden_sections = (".debug", ".nv_debug", ".ptx", ".ttir", ".ttgir", ".llvm")
    for name in section_names(cubin):
        lowered = name.lower()
        if any(token in lowered for token in forbidden_sections):
            raise BuildError(f"release cubin contains forbidden section {name}")
    if forbidden_source_path:
        normalized = forbidden_source_path.replace(b"\\", b"/")
        if forbidden_source_path in cubin or normalized in cubin:
            raise BuildError("release cubin embeds the AOT source path")
