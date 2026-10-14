"""Motor didáctico de orientación basado en reglas y Random Forest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

INTERESTS = {
    "tecnologia": "Tecnología y resolución de problemas",
    "ciencia": "Ciencia e investigación",
    "creatividad": "Creatividad y comunicación",
    "social": "Ayudar y trabajar con personas",
    "empresa": "Organización y emprendimiento",
    "exterior": "Naturaleza y trabajo al aire libre",
}

PATHS = {
    "Ingeniería y tecnología": {
        "center": (4.6, 3.8, 3.0, 2.5, 2.8, 2.0),
        "studies": ("Ingeniería informática", "Ciberseguridad", "Electrónica", "Telecomunicaciones"),
        "description": "Crear soluciones técnicas, sistemas y productos digitales.",
    },
    "Ciencias y datos": {
        "center": (3.2, 4.7, 2.7, 2.7, 2.5, 3.1),
        "studies": ("Ciencia de datos", "Matemáticas", "Biotecnología", "Investigación científica"),
        "description": "Investigar, analizar evidencias y encontrar patrones.",
    },
    "Salud y bienestar": {
        "center": (2.3, 3.8, 2.8, 4.7, 2.6, 2.0),
        "studies": ("Enfermería", "Fisioterapia", "Psicología", "Medicina"),
        "description": "Mejorar la salud y el bienestar de las personas.",
    },
    "Diseño y comunicación": {
        "center": (3.0, 2.8, 4.8, 3.4, 3.1, 2.2),
        "studies": ("Diseño gráfico", "Diseño audiovisual", "UX/UI", "Comunicación"),
        "description": "Comunicar ideas y diseñar experiencias o contenidos.",
    },
    "Empresa y gestión": {
        "center": (3.0, 2.7, 3.2, 3.4, 4.8, 2.3),
        "studies": ("Administración de empresas", "Marketing", "Finanzas", "Emprendimiento"),
        "description": "Organizar proyectos, equipos y recursos para alcanzar objetivos.",
    },
    "Medio ambiente": {
        "center": (2.8, 4.0, 2.8, 2.8, 2.4, 4.8),
        "studies": ("Ciencias ambientales", "Energías renovables", "Ecología", "Ingeniería forestal"),
        "description": "Proteger los ecosistemas y desarrollar soluciones sostenibles.",
    },
}


@dataclass(frozen=True)
class Rule:
    path: str
    interest: str
    minimum: int
    points: int
    explanation: str


@dataclass(frozen=True)
class RuleResult:
    path: str
    score: int
    evidence: tuple[str, ...]


RULES = (
    Rule("Ingeniería y tecnología", "tecnologia", 4, 3, "te atrae la tecnología"),
    Rule("Ingeniería y tecnología", "ciencia", 3, 1, "también te interesa la ciencia"),
    Rule("Ingeniería y tecnología", "creatividad", 3, 1, "te gusta crear soluciones"),
    Rule("Ciencias y datos", "ciencia", 4, 3, "destaca tu interés por investigar"),
    Rule("Ciencias y datos", "tecnologia", 3, 1, "te interesa analizar herramientas y datos"),
    Rule("Ciencias y datos", "exterior", 3, 1, "te atraen los temas del entorno natural"),
    Rule("Salud y bienestar", "social", 4, 3, "te motiva ayudar a otras personas"),
    Rule("Salud y bienestar", "ciencia", 3, 2, "te interesa comprender cómo funcionan las cosas"),
    Rule("Salud y bienestar", "creatividad", 3, 1, "valoras encontrar distintas maneras de ayudar"),
    Rule("Diseño y comunicación", "creatividad", 4, 3, "destaca tu creatividad"),
    Rule("Diseño y comunicación", "social", 3, 1, "te interesa comunicarte con otras personas"),
    Rule("Diseño y comunicación", "tecnologia", 3, 1, "te interesan las herramientas digitales"),
    Rule("Empresa y gestión", "empresa", 4, 3, "te atraen la organización y los proyectos"),
    Rule("Empresa y gestión", "social", 3, 1, "te interesa colaborar con otras personas"),
    Rule("Empresa y gestión", "tecnologia", 3, 1, "te interesan las soluciones prácticas"),
    Rule("Medio ambiente", "exterior", 4, 3, "te atrae trabajar con la naturaleza"),
    Rule("Medio ambiente", "ciencia", 3, 2, "te interesa comprender el entorno"),
    Rule("Medio ambiente", "tecnologia", 3, 1, "te interesan posibles soluciones técnicas"),
)


def validate_profile(profile: Mapping[str, int]) -> np.ndarray:
    """Valida y devuelve las respuestas en el orden de INTERESTS."""
    missing = set(INTERESTS).difference(profile)
    extra = set(profile).difference(INTERESTS)
    if missing or extra:
        raise ValueError(f"Perfil inválido. Faltan: {sorted(missing)}; sobran: {sorted(extra)}.")

    values: list[int] = []
    for key in INTERESTS:
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value <= 5:
            raise ValueError(f"La respuesta «{key}» debe ser un entero entre 0 y 5.")
        values.append(int(value))
    return np.asarray(values, dtype=np.float32)


def evaluate_expert_rules(profile: Mapping[str, int]) -> list[RuleResult]:
    """Aplica reglas IF/THEN y devuelve las rutas con mayor evidencia."""
    validate_profile(profile)
    scores = {path: 0 for path in PATHS}
    evidence: dict[str, list[str]] = {path: [] for path in PATHS}

    for rule in RULES:
        if profile[rule.interest] >= rule.minimum:
            scores[rule.path] += rule.points
            evidence[rule.path].append(rule.explanation)

    results = [
        RuleResult(path=path, score=scores[path], evidence=tuple(evidence[path]))
        for path in PATHS
        if scores[path] > 0
    ]
    return sorted(results, key=lambda result: (-result.score, result.path))


def train_demo_models(
    samples_per_path: int = 100,
) -> tuple[DecisionTreeClassifier, RandomForestClassifier]:
    """Entrena un árbol y un bosque reproducibles con perfiles simulados."""
    if samples_per_path < 2:
        raise ValueError("Se necesitan al menos dos perfiles simulados por ruta.")

    rng = np.random.default_rng(42)
    centers = np.asarray([PATHS[path]["center"] for path in PATHS], dtype=np.float32)
    labels = list(PATHS)
    samples = np.vstack(
        [
            np.clip(rng.normal(center, 0.9, size=(samples_per_path, len(INTERESTS))), 0, 5)
            for center in centers
        ]
    ).astype(np.float32)
    targets = np.repeat(labels, samples_per_path)

    decision_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    decision_tree.fit(samples, targets)

    forest = RandomForestClassifier(
        n_estimators=160,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(samples, targets)
    return decision_tree, forest


def predict_with_model(
    model: DecisionTreeClassifier | RandomForestClassifier,
    profile: Mapping[str, int],
) -> list[tuple[str, float]]:
    """Devuelve las rutas ordenadas por probabilidad estimada por el modelo."""
    features = validate_profile(profile).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    return sorted(
        (
            (str(path), float(probability))
            for path, probability in zip(model.classes_, probabilities)
        ),
        key=lambda item: (-item[1], item[0]),
    )
