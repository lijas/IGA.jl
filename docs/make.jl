using Documenter
using IGA

include("generate.jl")

GENERATEDEXAMPLES = [joinpath("examples", f) for f in (
    "plate_with_hole.md",
    "structuralvibrations.md"
    )]

makedocs(
    sitename = "IGA",
    format = Documenter.HTML(),
    doctest = false,
    #modules = [IGA],
    strict = false,
    pages = Any[
        "Home" => "index.md",
        "Manual" => ["bsplines_nurbs.md", "bezier_extraction.md"],
        "Examples" => GENERATEDEXAMPLES,
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
deploydocs(
    repo = "github.com/lijas/IGA.jl.git",
    push_preview=true,
    devbranch = "master"
)
