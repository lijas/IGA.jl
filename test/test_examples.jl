

module TestPlateWithHoleExample
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate/plate_with_hole.jl"))
        end
    end
end