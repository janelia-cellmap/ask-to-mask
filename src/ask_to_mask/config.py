"""Organelle class definitions: names, colors, and prompt templates."""

from dataclasses import dataclass, field


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

    def _build_context(
        self, detailed: bool = False, resolution_nm: float | None = None,
    ) -> list[str]:
        """Build context parts: EM preamble and optional description/resolution."""
        parts = ["This is an EM image of cell(s)."]
        if resolution_nm is not None:
            parts.append(f"The image resolution is {resolution_nm:.0f}nm/px.")
        if detailed and self.description:
            parts.append(f"In EM, {self.name} appear as: {self.description}")
        return parts

    def build_prompt(
        self, detailed: bool = False, resolution_nm: float | None = None,
    ) -> str:
        """Build a semantic segmentation prompt, optionally with EM description."""
        parts = self._build_context(detailed, resolution_nm=resolution_nm)
        parts.append(
            f"Create a segmentation mask: color all the {self.name} in "
            f"{self.color_name} and make everything else black."
        )
        return " ".join(parts)

    def build_prompt_varied(
        self, rng, resolution_nm: float | None = None,
    ) -> str:
        """Build a prompt using a randomly selected template for training."""
        parts = self._build_context(resolution_nm=resolution_nm)
        idx = rng.integers(len(PROMPT_TEMPLATES))
        parts.append(
            PROMPT_TEMPLATES[idx].format(name=self.name, color_name=self.color_name)
        )
        return " ".join(parts)

    def build_direct_prompt(
        self, detailed: bool = False, resolution_nm: float | None = None,
    ) -> str:
        """Build a direct mask prompt: white organelle on black background."""
        parts = self._build_context(detailed, resolution_nm=resolution_nm)
        parts.append(
            f"Create a segmentation mask: show only the {self.name} in white "
            f"on a completely black background. Nothing else should be visible."
        )
        return " ".join(parts)

    def build_invert_prompt(
        self, detailed: bool = False, resolution_nm: float | None = None,
    ) -> str:
        """Build a prompt that segments background/edges, to be inverted for instances."""
        parts = self._build_context(detailed, resolution_nm=resolution_nm)
        parts.append(
            f"Create a segmentation mask: color the background and edges between "
            f"{self.name} in white on a completely black background. The {self.name} "
            f"interiors should be black, only the boundaries and space between them "
            f"should be white."
        )
        return " ".join(parts)

    def build_instance_prompt(
        self, detailed: bool = False, resolution_nm: float | None = None,
        direct: bool = False,
    ) -> str:
        """Build an instance segmentation prompt, optionally with EM description.

        If direct=True, instances are colored on a black background instead of
        on the original EM image.
        """
        parts = self._build_context(detailed, resolution_nm=resolution_nm)
        if direct:
            parts.append(
                f"Color each individual {self.name} a different unique bright color "
                f"to distinguish separate instances. Make everything else completely black."
            )
        else:
            parts.append(
                f"Color each individual {self.name} a different unique bright color "
                f"to distinguish separate instances. Keep everything else unchanged."
            )
        return " ".join(parts)


# Alternative prompt templates for training variation.
# Each must contain {name} and {color_name} placeholders.
PROMPT_TEMPLATES = [
    "Color all the {name} in {color_name}. Keep everything else unchanged.",
    "Highlight the {name} in {color_name}. Do not change anything else.",
    "Paint all {name} {color_name}. Leave everything else as is.",
    "Mark the {name} with {color_name}. Keep the rest unchanged.",
    "Color every {name} {color_name}. Do not modify the background.",
]

# Alternative templates for multi-organelle prompts.
# Each must contain {instructions} placeholder.
MULTI_PROMPT_TEMPLATES = [
    "Color {instructions}. Keep everything else unchanged.",
    "Highlight {instructions}. Do not change anything else.",
    "Paint {instructions}. Leave everything else as is.",
    "Mark {instructions}. Keep the rest unchanged.",
]


def build_multi_organelle_prompt(
    organelles: list[OrganelleClass],
    resolution_nm: float | None = None,
    rng=None,
) -> str:
    """Build a prompt that asks the model to color multiple organelles at once."""
    parts = ["This is an EM image of cell(s)."]
    if resolution_nm is not None:
        parts.append(f"The image resolution is {resolution_nm:.0f}nm/px.")
    color_instructions = [
        f"the {org.name} in {org.color_name}" for org in organelles
    ]
    if len(color_instructions) == 1:
        joined = color_instructions[0]
    else:
        joined = ", ".join(color_instructions[:-1]) + " and " + color_instructions[-1]
    if rng is not None:
        idx = rng.integers(len(MULTI_PROMPT_TEMPLATES))
        parts.append(MULTI_PROMPT_TEMPLATES[idx].format(instructions=joined))
    else:
        parts.append(f"Color {joined}. Keep everything else unchanged.")
    return " ".join(parts)


ORGANELLES: dict[str, OrganelleClass] = {
    "mito": OrganelleClass(
        key="mito",
        name="mitochondria",
        color_name="bright red",
        rgb=(255, 0, 0),
        description=(
            "large (500-2000 nm), ovoid or elongated organelles with a smooth outer "
            "membrane and a highly folded inner membrane forming cristae (visible as "
            "dark internal lamellar or tubular folds). They appear as dark-bordered "
            "oval or bean-shaped structures with a distinctive internal striated or "
            "layered texture. They can fuse and branch to form tubular networks. "
            "Their electron density is moderate—darker than cytosol but lighter than "
            "ribosomes. Cristae density varies by cell type and metabolic state."
        ),
    ),
    "er": OrganelleClass(
        key="er",
        name="endoplasmic reticulum",
        color_name="bright green",
        rgb=(0, 255, 0),
        description=(
            "an extensive, interconnected network of flattened cisternae and tubules "
            "(30-60 nm lumen width) that pervades the cytoplasm. Rough ER appears as "
            "parallel membrane pairs studded with dark ribosome dots on the cytoplasmic "
            "face; smooth ER lacks ribosomes and forms more tubular profiles. ER tubules "
            "always connect back to themselves and ultimately connect to the nuclear "
            "envelope. Unlike similar-looking multivesicular bodies or vesicles, ER is "
            "never disconnected—it forms a single continuous network."
        ),
    ),
    "nucleus": OrganelleClass(
        key="nucleus",
        name="nucleus",
        color_name="bright blue",
        rgb=(0, 0, 255),
        description=(
            "the largest organelle (5-15 µm diameter), roughly spherical or ovoid, "
            "bounded by a double-membrane nuclear envelope perforated with nuclear "
            "pores. The interior contains chromatin: dark, electron-dense heterochromatin "
            "clusters (often along the envelope periphery) and lighter, diffuse "
            "euchromatin. A dense, spherical nucleolus (1-5 µm) is often visible as "
            "the darkest sub-nuclear structure. The nucleoplasm between chromatin "
            "regions appears as a relatively uniform, light gray matrix."
        ),
    ),
    "lipid_droplet": OrganelleClass(
        key="lipid_droplet",
        name="lipid droplets",
        color_name="bright yellow",
        rgb=(255, 255, 0),
        description=(
            "spherical organelles (100 nm-5 µm) enclosed by a lipid monolayer (not a "
            "bilayer). In heavy-metal stained EM they appear with a shriveled, lumpy "
            "morphology and a generally lighter, more homogeneous interior compared to "
            "the surrounding cytosol. Their boundary shows subtle membrane staining. "
            "Distinguished from lysosomes and endosomes by their lighter, more uniform "
            "interior and absence of internal vesicles or dense granular content."
        ),
    ),
    "plasma_membrane": OrganelleClass(
        key="plasma_membrane",
        name="plasma membrane",
        color_name="bright cyan",
        rgb=(0, 255, 255),
        description=(
            "a thin (~7-8 nm) dark line delineating the outer boundary of the cell, "
            "separating extracellular space from the cytosol. It appears as a continuous "
            "electron-dense bilayer that follows the entire cell contour, including "
            "microvilli, filopodia, and invaginations. It always forms a closed boundary "
            "around a cell. If a membrane segment appears disconnected from the cell "
            "surface, it is likely part of the endosomal network instead."
        ),
    ),
    "nuclear_envelope": OrganelleClass(
        key="nuclear_envelope",
        name="nuclear envelope",
        color_name="bright magenta",
        rgb=(255, 0, 255),
        description=(
            "a double-membrane structure (~30-50 nm total thickness) consisting of two "
            "parallel lipid bilayers (inner and outer nuclear membranes) separated by "
            "the perinuclear space. The outer membrane is often studded with ribosomes, "
            "making it continuous with rough ER. It forms a closed boundary around the "
            "nucleus and is perforated by ~120 nm nuclear pore complexes visible as "
            "gaps or electron-dense ring structures. It separates the chromatin-containing "
            "nucleoplasm from the cytosol."
        ),
    ),
    "nuclear_pore": OrganelleClass(
        key="nuclear_pore",
        name="nuclear pores",
        color_name="bright orange",
        rgb=(255, 128, 0),
        description=(
            "~120 nm diameter protein complexes that perforate the nuclear envelope. "
            "In en-face views they appear as ring-shaped electron-dense structures; in "
            "cross-section they appear as breaks or gaps in envelope connectivity where "
            "the inner and outer nuclear membranes fuse. A central plug or transporter "
            "is sometimes visible. They span both bilayers of the nuclear envelope and "
            "are distributed across the entire nuclear surface."
        ),
    ),
    "nucleolus": OrganelleClass(
        key="nucleolus",
        name="nucleolus",
        color_name="bright purple",
        rgb=(128, 0, 255),
        description=(
            "a large (1-5 µm), dense, roughly spherical sub-nuclear body that is the "
            "most electron-dense structure within the nucleus. It has a granular and "
            "fibrillar internal texture with distinct sub-compartments: a dense fibrillar "
            "component, a granular component, and sometimes fibrillar centers (lighter "
            "regions). It stains significantly darker than surrounding chromatin and "
            "nucleoplasm. Not bounded by a membrane."
        ),
    ),
    "heterochromatin": OrganelleClass(
        key="heterochromatin",
        name="heterochromatin",
        color_name="bright spring green",
        rgb=(0, 255, 128),
        description=(
            "dark, electron-dense, compact clusters of tightly packed chromatin within "
            "the nucleus. Typically found along the inner surface of the nuclear envelope "
            "(peripheral heterochromatin) and around the nucleolus (perinucleolar "
            "heterochromatin). Stains significantly darker and is more compact than the "
            "diffuse euchromatin regions. Excludes the nucleolus itself, which has a "
            "distinct granular/fibrillar texture."
        ),
    ),
    "euchromatin": OrganelleClass(
        key="euchromatin",
        name="euchromatin",
        color_name="bright rose",
        rgb=(255, 0, 128),
        description=(
            "light, diffuse, loosely packed chromatin that fills much of the nuclear "
            "interior between heterochromatin clusters. Appears as a lighter gray, more "
            "uniform matrix compared to the dark heterochromatin. Represents "
            "transcriptionally active chromatin regions. Located outside of the nucleolus "
            "and distinct from the dense nucleolar sub-structure. Excludes chromatin "
            "directly associated with the nucleolus."
        ),
    ),
    "cell": OrganelleClass(
        key="cell",
        name="cells",
        color_name="bright red",
        rgb=(255, 0, 0),
        description=(
            "whole cells bounded by a plasma membrane. Each cell contains cytoplasm "
            "filled with various organelles (mitochondria, ER, nucleus, etc.) and a "
            "relatively uniform cytosolic matrix. Cell boundaries are defined by the "
            "plasma membrane—a thin dark line separating one cell from another or from "
            "extracellular space. Extracellular regions may appear lighter or contain "
            "extracellular matrix material."
        ),
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
