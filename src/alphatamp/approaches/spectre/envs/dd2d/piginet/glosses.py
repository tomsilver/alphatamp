"""DD2D domain vocabulary → colloquial NL glosses for the CLIP-text encoder (paper
§IV-A).

PIGINet feeds each domain word (operator / predicate / constant) through a frozen language
model as a short English phrase rather than its raw token ("rephrasing helps the network
deal with out-of-distribution names like ``isjointto``"). We keep a fixed table for the
whole DD2D vocabulary (verified present across the collected train split).

Words that appear as *object arguments* (``target``, ``o0`` …) are NOT glossed here — they
are encoded by the image+geometry object channel (``encoders.object_feat``), not by text.
The glossed words are the operators, predicates, and the object attributes (category / color
/ family / region) available for the object-channel ablation.
"""

from __future__ import annotations

GLOSSES: dict[str, str] = {
    # operators (task-plan actions)
    "pick": "grasp and lift an object out of the drawer",
    "place-buffer": "place the carried object onto the staging buffer area beside the drawer",
    "retrieve": "grasp and remove the target object from the drawer",
    # predicates (init / goal literals)
    "handempty": "the robot gripper is empty and holding nothing",
    "in-drawer": "an object is resting inside the drawer",
    "target": "the target object that must be extracted",
    "extracted": "the target object has been removed from the drawer",
    "at-pose": "an object is located at this position and orientation",
    # object categories
    "item": "an ordinary household item cluttering the drawer",
    # object colors (also encode the concave/convex distinction in our render)
    "tomato": "a red target object",
    "slateblue": "a blue concave-shaped item",
    "silver": "a grey convex item",
    # shape families
    "can": "a small round can",
    "bowl": "a large round bowl",
    "box": "a rectangular box",
    "pillcase": "a long rounded capsule-shaped case",
    "dumbbell": "a dumbbell with two ends and a narrow waist",
    "shoe": "an L-shaped shoe with a concave corner",
    "horseshoe": "a blocky C-shaped horseshoe with two prongs",
    # regions
    "drawer": "the drawer interior holding the clutter",
    "buffer": "the staging buffer area beside the drawer",
}

VOCAB: list[str] = sorted(GLOSSES)


def gloss(word: str) -> str:
    """NL phrase for a domain word (falls back to the word itself if unglossed)."""
    return GLOSSES.get(word, word)
