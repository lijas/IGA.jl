

using Ferrite
using IGA

function _get_face_bezier_coordinates!(xb::Vector{<:Vec{dim}}, x::Vector{<:Vec{dim}}, grid::BezierGrid, cell::Ferrite.AbstractCell, lfaceid::Int, cellid::Int) where dim
    ip = Ferrite.default_interpolation(typeof(cell))
    
    facedofs = Ferrite.dirichlet_facedof_indices(ip)[lfaceid]
    @assert length(x) == length(facedofs)
    @assert length(x) == length(xb)

    i = 0
    for d in facedofs
        i += 1
        x[i] = grid.nodes[cell.nodes[d]].x
    end

    n = length(xb)
	C = grid.beo[cellid]

    for i in 1:n
        xb[i] = zero(Vec{dim})
    end

    for i in 1:n
        k = facedofs[i]
		c_row = C[k]
		_x = x[i]
		for j in 1:n
            dof = facedofs[j]
            val = c_row[dof]              
			xb[j] += val * _x
		end
	end

end

ip = IGA.IGAInterpolation{RefQuadrilateral,2}()
ipface = IGA.IGAInterpolation{RefLine,2}()

nurbsmesh = generate_nurbs_patch(:cube, (3,3), (2,2); size = (1.0, 1.0))
grid = BezierGrid(nurbsmesh)

dh = DofHandler(grid)
add!(dh, :u, ip^2)
close!(dh)

cellid = 3
lfaceid = 2
cell = grid.cells[cellid]

a = zeros(ndofs(dh)) .+1

vtk = VTKIGAFile("newoutput", grid, 1:getncells(grid))
IGA.write_solution(vtk, dh, a)
close(vtk)

n = 3
xb = zeros(Vec{2}, n)
x = zeros(Vec{2}, n)

N = zeros(n)
dNdξ = zeros(Vec{1}, n)
ξ = zero(Vec{1})
for i in 1:n
    dNdξ[i], N[i] = Ferrite.shape_gradient_and_value(ipface, ξ, i)
end

_get_face_bezier_coordinates!(xb, x, grid, cell, lfaceid, cellid)

dxdxi = zero(Vec{2})
for i in 1:n
    dxdxi += dNdξ[i][1] * xb[i]
end

_normal(dxdxi) = Vec((dxdxi[2], -dxdxi[1]))/norm( Vec((dxdxi[2], -dxdxi[1])))

@show _normal(dxdxi)

