"""Procedurally generated product manuals (long docs) with gold fact spans, plus paraphrased questions.

Every manual has 7 sections. Six facts per product are planted at random positions inside the
filler text, and we record the exact character span of each fact sentence and of its answer string.
Fact sentences say "the unit" and never name the product, as real manuals often do. Only the title
and some filler sentences carry the product name, so chunks lose context when they are cut away from it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

PRODUCTS = ["Aurora X2 drone", "Brightline kettle", "Cobalt S7 speaker", "Delta Pro router",
            "Ember smart lamp", "Falcon GPS watch", "Glacier air purifier", "Harbor e-reader"]

FILLER = {
    "Overview": ["The {P} is designed for everyday use at home and on the road.",
                 "This guide explains how to get the most out of your {P}.",
                 "Please read every section before first use.",
                 "The package includes the main unit, a cable, and a quick start card.",
                 "Firmware updates are delivered automatically through the companion app.",
                 "Keep this manual for future reference."],
    "Battery": ["The battery is a sealed lithium-ion pack that cannot be replaced by the user.",
                "Battery health gradually declines after several hundred charge cycles.",
                "The indicator turns green when charging is complete.",
                "Avoid leaving the {P} in a hot car while it charges.",
                "Low power mode extends runtime by dimming the display.",
                "Only use certified cables to protect the charging port."],
    "Setup": ["Download the companion app and create an account.",
              "Turn the unit on and wait for the pairing tone.",
              "Select your home network and enter the password.",
              "The first sync can take several minutes.",
              "Place the {P} on a flat, stable surface during setup.",
              "You can rename the device in the app settings."],
    "Maintenance": ["Dust and fingerprints do not affect performance but may look untidy.",
                    "Check the vents on the {P} regularly and keep them free of lint.",
                    "Do not use abrasive powders or solvents.",
                    "Store the device in a dry place when not in use.",
                    "Inspect the cable for fraying every few months.",
                    "Let the unit dry completely before switching it back on."],
    "Specifications": ["The housing is made from recycled aluminium and polycarbonate.",
                       "Wireless range is up to thirty metres in open space.",
                       "The {P} weighs less than most devices in its class.",
                       "Humidity above ninety percent may cause condensation inside.",
                       "The display is readable in direct sunlight.",
                       "Running outside the rated range may shut the unit down to protect it."],
    "Warranty": ["Keep your proof of purchase to make a claim.",
                 "Damage from drops, water, or unauthorised repairs is not covered.",
                 "Claims for the {P} are handled through the support portal.",
                 "Replacement units carry the remainder of the original coverage.",
                 "Extended protection plans can be purchased separately.",
                 "Consumer rights under local law are not affected."],
    "Troubleshooting": ["If the unit does not respond, first try charging it for thirty minutes.",
                        "A flashing red light usually means the battery is critically low.",
                        "Restarting the companion app solves most connection problems with the {P}.",
                        "Resetting erases all settings and paired accounts.",
                        "After a reset you will need to repeat the setup steps.",
                        "Contact support if the problem persists after a reset."],
}

# fact type -> (section, value generator, sentence templates, paraphrased question templates)
FACTS = {
    "battery": ("Battery", lambda r: f"{r.randint(6, 30)} hours",
                ["On a full charge the battery lasts {v} of continuous use.",
                 "Expect roughly {v} of runtime from a single charge."],
                ["How long does the {P} run on one charge?", "What battery life should I expect from the {P}?"]),
    "charging": ("Battery", lambda r: f"{r.randrange(35, 180, 5)} minutes",
                 ["A complete recharge takes about {v} with the supplied USB-C adapter.",
                  "Refilling an empty battery to full needs around {v} on the included adapter."],
                 ["How long does it take to fully recharge the {P}?", "How much time does the {P} need to charge from empty?"]),
    "reset": ("Troubleshooting",
              lambda r: f"hold the {r.choice(['power', 'pairing', 'mode', 'rear'])} button for {r.randint(5, 20)} seconds",
              ["To perform a factory reset, {v} until the status light blinks white.",
               "Restoring factory settings is simple: {v} and then release it."],
              ["How do I factory reset the {P}?", "What are the steps to restore the {P} to factory settings?"]),
    "temperature": ("Specifications", lambda r: (lambda a: f"between {a} and {a + r.randint(30, 45)} degrees Celsius")(r.randint(-20, 5)),
                    ["The unit is rated to operate {v}.",
                     "Reliable operation is guaranteed only at ambient temperatures {v}."],
                    ["What temperature range can the {P} operate in?", "How cold or hot can it get before the {P} stops working?"]),
    "warranty": ("Warranty", lambda r: f"{r.randint(1, 5)}-year limited warranty",
                 ["Every unit ships with a {v} covering manufacturing defects.",
                  "Manufacturing defects are covered by a {v} from the date of purchase."],
                 ["How many years of warranty coverage does the {P} have?", "What warranty comes with the {P}?"]),
    "cleaning": ("Maintenance",
                 lambda r: f"wipe the {r.choice(['outer shell', 'front grille', 'top panel', 'sensor window'])} with a {r.choice(['dry microfibre cloth', 'slightly damp cotton cloth', 'soft lint-free cloth'])}",
                 ["For routine cleaning, {v} once a week and never submerge the unit.",
                  "To keep it looking new, {v}; do not rinse it under a tap."],
                 ["What is the recommended way to clean the {P}?", "How should I clean the {P} safely?"]),
}
SECTIONS = ["Overview", "Battery", "Setup", "Maintenance", "Specifications", "Warranty", "Troubleshooting"]


@dataclass
class Doc:
    id: str
    title: str
    text: str
    facts: Dict[str, dict] = field(default_factory=dict)  # fact type -> {answer, fact_span, answer_span}


@dataclass
class Query:
    id: str
    doc_id: str
    fact: str
    text: str
    answer: str


def build_corpus(seed: int = 42) -> Tuple[List[Doc], List[Query]]:
    r = random.Random(seed)
    docs, queries = [], []
    for pi, P in enumerate(PRODUCTS):
        did = f"d{pi}"
        text = f"# {P.title()} user manual"
        facts = {}
        for sec in SECTIONS:
            sents = [(s.format(P=P), None) for s in r.sample(FILLER[sec], r.randint(4, 5))]
            for ft, (fsec, gen, tmpls, _) in FACTS.items():
                if fsec == sec:
                    v = gen(r)
                    sents.insert(r.randint(0, len(sents)), (r.choice(tmpls).format(v=v), (ft, v)))
            split = len(sents) // 2
            text += f"\n\n## {sec}\n"
            for i, (s, meta) in enumerate(sents):
                if i == split:
                    text += "\n\n"
                elif i > 0:
                    text += " "
                start = len(text)
                text += s
                if meta:
                    ft, v = meta
                    a0 = text.index(v, start)
                    facts[ft] = {"answer": v, "fact_span": (start, len(text)), "answer_span": (a0, a0 + len(v))}
        text += "\n"
        docs.append(Doc(did, P, text, facts))
        for ft, (_, _, _, qt) in FACTS.items():
            queries.append(Query(f"{did}-{ft}", did, ft, r.choice(qt).format(P=P), facts[ft]["answer"]))
    return docs, queries
