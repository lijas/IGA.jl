using Documenter
using IGA

include("generate.jl")

GENERATEDEXAMPLES = [joinpath("examples", f) for f in (
    "example1.md",
    )]

makedocs(
    sitename = "IGA",
    format = Documenter.HTML(),
    doctest = false,
    #modules = [IGA],
    strict = false,
    pages = Any[
        "Home" => "index.md",
        "Examples" => GENERATEDEXAMPLES,
        "Iga2" => "Isogeom.md",
        "Someelse" => ["someelse/index.md", "someelse/Isogeom.md"],
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
