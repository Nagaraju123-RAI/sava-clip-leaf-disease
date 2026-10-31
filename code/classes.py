"""Class metadata for PlantVillage (38 classes) with symptom descriptions,
and the mapping from PlantDoc folder names to PlantVillage classes.

Each entry: folder name -> (crop, condition, symptom description).
Descriptions are short visual phrases written for prompting a
vision-language model; they describe what the symptom LOOKS like.
"""

CLASSES = {
    "Apple___Apple_scab": ("apple", "apple scab",
        "olive-green to dark brown velvety spots that become corky lesions on the leaf"),
    "Apple___Black_rot": ("apple", "black rot",
        "small purple spots that enlarge into brown frog-eye lesions with tan centers and purple margins"),
    "Apple___Cedar_apple_rust": ("apple", "cedar apple rust",
        "bright yellow-orange spots with a reddish border and tiny dark dots in the center"),
    "Apple___healthy": ("apple", "healthy",
        "a uniformly green apple leaf with a smooth surface and no spots"),
    "Blueberry___healthy": ("blueberry", "healthy",
        "a smooth oval uniformly green blueberry leaf without lesions"),
    "Cherry_(including_sour)___Powdery_mildew": ("cherry", "powdery mildew",
        "white powdery fungal patches on the leaf surface with slight curling"),
    "Cherry_(including_sour)___healthy": ("cherry", "healthy",
        "a glossy uniformly green cherry leaf with serrated edges and no spots"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("corn", "gray leaf spot",
        "long narrow rectangular gray to tan lesions running parallel between the veins"),
    "Corn_(maize)___Common_rust_": ("corn", "common rust",
        "many small elongated cinnamon-brown powdery pustules scattered across the blade"),
    "Corn_(maize)___Northern_Leaf_Blight": ("corn", "northern leaf blight",
        "long elliptical cigar-shaped gray-green to tan lesions on the blade"),
    "Corn_(maize)___healthy": ("corn", "healthy",
        "a long green corn blade with clean parallel veins and no lesions"),
    "Grape___Black_rot": ("grape", "black rot",
        "circular reddish-brown spots with dark borders and tiny black dots inside"),
    "Grape___Esca_(Black_Measles)": ("grape", "esca (black measles)",
        "tiger-stripe pattern of yellow or red bands between veins with dried brown stripes"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("grape", "leaf blight",
        "irregular dark brown to black angular spots with drying leaf tissue"),
    "Grape___healthy": ("grape", "healthy",
        "a lobed uniformly green grape leaf without spots or discoloration"),
    "Orange___Haunglongbing_(Citrus_greening)": ("orange", "citrus greening",
        "blotchy asymmetric yellow mottling of the leaf with yellowed veins"),
    "Peach___Bacterial_spot": ("peach", "bacterial spot",
        "small angular purple-brown spots that may fall out leaving shot holes"),
    "Peach___healthy": ("peach", "healthy",
        "a narrow lance-shaped uniformly green peach leaf"),
    "Pepper,_bell___Bacterial_spot": ("bell pepper", "bacterial spot",
        "small water-soaked spots turning brown with yellow halos on the leaf"),
    "Pepper,_bell___healthy": ("bell pepper", "healthy",
        "a smooth uniformly green bell pepper leaf with no spots"),
    "Potato___Early_blight": ("potato", "early blight",
        "dark brown circular spots with concentric target-like rings and yellow surroundings"),
    "Potato___Late_blight": ("potato", "late blight",
        "large irregular water-soaked dark brown to black lesions with pale margins"),
    "Potato___healthy": ("potato", "healthy",
        "a uniformly green potato leaflet without lesions"),
    "Raspberry___healthy": ("raspberry", "healthy",
        "a serrated uniformly green raspberry leaflet without spots"),
    "Soybean___healthy": ("soybean", "healthy",
        "an oval uniformly green soybean leaflet without spots"),
    "Squash___Powdery_mildew": ("squash", "powdery mildew",
        "white powdery spots and coating spread over the leaf surface"),
    "Strawberry___Leaf_scorch": ("strawberry", "leaf scorch",
        "many small irregular dark purple spots with dry scorched leaf edges"),
    "Strawberry___healthy": ("strawberry", "healthy",
        "a toothed uniformly green strawberry leaflet without spots"),
    "Tomato___Bacterial_spot": ("tomato", "bacterial spot",
        "small dark greasy brown spots, sometimes with yellow halos"),
    "Tomato___Early_blight": ("tomato", "early blight",
        "brown spots with concentric rings surrounded by yellowing tissue"),
    "Tomato___Late_blight": ("tomato", "late blight",
        "large irregular greasy gray-green to brown lesions spreading across the leaf"),
    "Tomato___Leaf_Mold": ("tomato", "leaf mold",
        "pale yellow patches on the upper surface with olive-green velvety mold beneath"),
    "Tomato___Septoria_leaf_spot": ("tomato", "septoria leaf spot",
        "many small circular spots with dark brown margins and gray-white centers"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("tomato", "spider mite damage",
        "fine yellow stippling and bronzing of the leaf, sometimes with fine webbing"),
    "Tomato___Target_Spot": ("tomato", "target spot",
        "brown lesions with concentric rings and light brown centers"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("tomato", "yellow leaf curl virus",
        "small upward-curled crumpled leaves with yellow margins"),
    "Tomato___Tomato_mosaic_virus": ("tomato", "mosaic virus",
        "light and dark green mottled mosaic pattern with leaf distortion"),
    "Tomato___healthy": ("tomato", "healthy",
        "a uniformly green compound tomato leaf with no spots"),
}

CLASS_NAMES = list(CLASSES.keys())          # fixed order = label index
CROPS = sorted({v[0] for v in CLASSES.values()})
CROP_OF_CLASS = [CROPS.index(CLASSES[c][0]) for c in CLASS_NAMES]

# PlantDoc folder -> PlantVillage class (all 27 PlantDoc classes have a counterpart)
PLANTDOC_MAP = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}

# Prompt templates used by the symptom-aware prompt ensemble (SAPE)
NAME_TEMPLATE = "a photo of a {crop} leaf with {cond}."
HEALTHY_NAME_TEMPLATE = "a photo of a healthy {crop} leaf."
SAPE_TEMPLATES = [
    "a photo of a {crop} leaf with {cond}, showing {desc}.",
    "a close-up photo of a {crop} leaf affected by {cond}: {desc}.",
    "a plant pathology image of {cond} on {crop}, with {desc}.",
    "a leaf of a {crop} plant diagnosed with {cond}, which shows {desc}.",
    "{desc}, a typical sign of {cond} on a {crop} leaf.",
]
SAPE_HEALTHY_TEMPLATES = [
    "a photo of a healthy {crop} leaf, {desc}.",
    "a close-up photo of a healthy {crop} leaf: {desc}.",
    "a plant pathology image of a healthy {crop} leaf with no disease.",
    "{desc}, a typical healthy {crop} leaf.",
    "a disease-free {crop} leaf.",
]


def name_prompts():
    out = []
    for c in CLASS_NAMES:
        crop, cond, _ = CLASSES[c]
        t = HEALTHY_NAME_TEMPLATE if cond == "healthy" else NAME_TEMPLATE
        out.append([t.format(crop=crop, cond=cond)])
    return out


def sape_prompts():
    out = []
    for c in CLASS_NAMES:
        crop, cond, desc = CLASSES[c]
        ts = SAPE_HEALTHY_TEMPLATES if cond == "healthy" else SAPE_TEMPLATES
        out.append([t.format(crop=crop, cond=cond, desc=desc) for t in ts])
    return out


def display_name(c):
    crop, cond, _ = CLASSES[c]
    return f"{crop} - {cond}"
