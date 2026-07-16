import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


# Tier 1: everything helpers.py's FRTD embedding itself needs.
CORE_REQUIRES = [
    "numpy",
    "scipy",
    "networkx",
    "tqdm",
]

# Tier 2: optional applications built on top of the core embedding.
EXTRAS_REQUIRE = {
    "role_extraction": [
        "pandas",
        "scikit-learn",
        "matplotlib",
        "python-louvain",  # imported as `community`
    ],
    "alignment": [
        "torch",  # FUGAL / FUGAL-FRTD Sinkhorn solver (Fugal-FRT/helpers)
    ],
    "randomisation": [
        "pathos",  # parallel MCMC replicas in graph_randomisation.py
    ],
    "distances": [
        "POT",  # imported as `ot`; only helpers.earth_movers_distance
    ],
    "benchmarks": [
        "pandas",
        "matplotlib",
        "scikit-learn",
        "gensim>=4.0.0",
        "joblib",
        "fastdtw",
        # struc2vec baseline; not on PyPI. Alternatively install the vendored
        # copy editable: pip install -e role_benchmarks/GraphEmbedding
        "ge @ git+https://github.com/shenweichen/GraphEmbedding.git",
    ],
}
EXTRAS_REQUIRE["all"] = sorted(set(
    req for reqs in EXTRAS_REQUIRE.values() for req in reqs
))

setuptools.setup(
    name="randomwalk-frtd-embeddings",
    version="0.1.0",
    author="Vedanta Thapar",
    author_email="vedantathapar@gmail.com",
    description="First-return-time distribution (FRTD) node embeddings for "
                 "graph role extraction, alignment, and randomisation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vedanta-T/RandomWalk_FRTD_embeddings",
    # Flat top-level modules rather than a package: Fugal-FRT/ and
    # role_benchmarks/ are used as standalone scripts (via sys.path/relative
    # imports), not importable subpackages, so they're intentionally left
    # out of py_modules and only covered by the extras above.
    py_modules=[
        "helpers",
        "graph_role_extraction",
        "graph_alignment",
        "graph_randomisation",
    ],
    python_requires=">=3.9",
    install_requires=CORE_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    license="MIT license",
)
