"""Organelle class definitions: names, colors, and prompt templates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OrganelleClass:
    key: str
    name: str
    color_name: str
    rgb: tuple[int, int, int]
    description: str = ""
    prompt_template: str = (
        "Color all the {name} in {color_name}. Keep everything else unchanged."
    )
    instance_prompt_template: str = (
        "Color each individual {name} a different unique bright color "
        "to distinguish separate instances. Keep everything else unchanged."
    )

    def _build_context(self, detailed: bool = False) -> list[str]:
        """Build context parts: EM preamble and optional description."""
        parts = ["This is an EM image of cell(s)."]
        if detailed and self.description:
            parts.append(f"In EM, {self.name} appear as: {self.description}")
        return parts

    def build_prompt(self, detailed: bool = False) -> str:
        """Build a semantic segmentation prompt, optionally with EM description."""
        parts = self._build_context(detailed)
        parts.append(
            f"Color all the {self.name} in {self.color_name}. "
            f"Keep everything else unchanged."
        )
        return " ".join(parts)

    def build_instance_prompt(self, detailed: bool = False) -> str:
        """Build an instance segmentation prompt, optionally with EM description."""
        parts = self._build_context(detailed)
        parts.append(
            f"Color each individual {self.name} a different unique bright color "
            f"to distinguish separate instances. Keep everything else unchanged."
        )
        return " ".join(parts)


ORGANELLES: dict[str, OrganelleClass] = {
    "mito": OrganelleClass(
        key="mito",
        name="mitochondria",
        color_name="bright red",
        rgb=(255, 0, 0),
        description=(
            "large, ovoid organelles with outer and inner membranes that form "
            "cristae. They can fuse and branch to form tubular networks. Inner "
            "membrane folding density varies by cell type."
        ),

    ),
    "er": OrganelleClass(
        key="er",
        name="endoplasmic reticulum",
        color_name="bright green",
        rgb=(0, 255, 0),
        description=(
            "an extensive network of tubular structures often studded with "
            "ribosomes. ER tubules always connect back to themselves and "
            "ultimately connect to the nuclear envelope. Unlike similar-looking "
            "multivesicular bodies, ER is never disconnected."
        ),

    ),
    "nucleus": OrganelleClass(
        key="nucleus",
        name="nucleus",
        color_name="bright blue",
        rgb=(0, 0, 255),
        description=(
            "a large, roughly spherical organelle bounded by a double-membrane "
            "nuclear envelope perforated with nuclear pores. Contains chromatin "
            "(dark heterochromatin clusters and lighter euchromatin) and often "
            "a dense, spherical nucleolus."
        ),
        # approx_size_nm=(5000, 15000),
    ),
    "lipid_droplet": OrganelleClass(
        key="lipid_droplet",
        name="lipid droplets",
        color_name="bright yellow",
        rgb=(255, 255, 0),
        description=(
            "spherical organelles enclosed by a lipid monolayer with a shriveled, "
            "'lumpy' morphology due to general staining. Generally lighter than "
            "surrounding cytosol with subtle membrane staining. Distinct texture "
            "compared to lysosomes and endosomes."
        ),
        # approx_size_nm=(100, 5000),
    ),
    "plasma_membrane": OrganelleClass(
        key="plasma_membrane",
        name="plasma membrane",
        color_name="bright cyan",
        rgb=(0, 255, 255),
        description=(
            "a thin, extensive lipid bilayer that surrounds the cell and "
            "separates extracellular space from the cytosol. Always connects "
            "back to itself. If disconnected, it is likely part of the "
            "endosomal network."
        ),
        # approx_size_nm=(7, 8),
    ),
    "nuclear_envelope": OrganelleClass(
        key="nuclear_envelope",
        name="nuclear envelope",
        color_name="bright magenta",
        rgb=(255, 0, 255),
        description=(
            "a double-membrane structure (two lipid bilayers) often studded "
            "with ribosomes. Connects back to itself and is continuous with "
            "ER networks. Perforated with ~120 nm nuclear pores. Establishes "
            "a boundary between chromatin and the cytosol."
        ),
        # approx_size_nm=(30, 50),
    ),
    "nuclear_pore": OrganelleClass(
        key="nuclear_pore",
        name="nuclear pores",
        color_name="bright orange",
        rgb=(255, 128, 0),
        description=(
            "circular, ~120 nm pores in the nuclear envelope. In cross-section, "
            "they appear as breaks or gaps in envelope connectivity. They span "
            "both bilayers of the nuclear envelope."
        ),
        # approx_size_nm=(100, 140),
    ),
    "nucleolus": OrganelleClass(
        key="nucleolus",
        name="nucleolus",
        color_name="bright purple",
        rgb=(128, 0, 255),
        description=(
            "a dark, dense spherical structure within the nucleus. Darker "
            "stained (denser) than the rest of the nucleus."
        ),
        # approx_size_nm=(1000, 5000),
    ),
    "heterochromatin": OrganelleClass(
        key="heterochromatin",
        name="heterochromatin",
        color_name="bright spring green",
        rgb=(0, 255, 128),
        description=(
            "dark, compact clusters of chromatin within the nucleus. Stains "
            "darker and is more compact than euchromatin. Excludes chromatin "
            "associated with the nucleolus."
        ),
        # approx_size_nm=(100, 5000),
    ),
    "euchromatin": OrganelleClass(
        key="euchromatin",
        name="euchromatin",
        color_name="bright rose",
        rgb=(255, 0, 128),
        description=(
            "light, diffuse chromatin within the nucleus. Stains lighter and "
            "is less compact than heterochromatin. Excludes chromatin "
            "associated with the nucleolus. Located outside of the nucleolus "
            "region."
        ),
        # approx_size_nm=(100, 10000),
    ),
}

# Mapping from ask-to-mask organelle keys to CellMap zarr fine-class directory names.
# Used by the training dataset to union fine-class labels into organelle-level masks.
ORGANELLE_FINE_CLASSES: dict[str, list[str]] = {
    "mito": ["mito_mem", "mito_lum", "mito_ribo"],
    "er": ["er_mem", "er_lum", "eres_mem", "eres_lum"],
    "nucleus": ["ne_mem", "ne_lum", "np_out", "np_in", "hchrom", "echrom", "nucpl"],
    "lipid_droplet": ["ld_mem", "ld_lum"],
    "plasma_membrane": ["pm"],
    "nuclear_envelope": ["ne_mem", "ne_lum"],
    "nuclear_pore": ["np_out", "np_in"],
    "nucleolus": [],
    "heterochromatin": ["hchrom"],
    "euchromatin": ["echrom"],
}

# Supported Flux model identifiers
MODELS = {
    "kontext-dev": "black-forest-labs/FLUX.1-Kontext-dev",
    "flux2-dev": "black-forest-labs/FLUX.2-dev",
}

DEFAULT_MODEL = "kontext-dev"
