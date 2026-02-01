"""STEP to OBJ/FBX conversion logic using Open Cascade XDE."""

from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import time

from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_SHELL, TopAbs_COMPOUND
from OCP.TopLoc import TopLoc_Location
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool, XCAFDoc_ColorSurf, XCAFDoc_ColorGen, XCAFDoc_ColorCurv
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_LabelSequence
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB


class OutputFormat(Enum):
    """Supported output formats."""
    OBJ = "obj"
    FBX = "fbx"


@dataclass
class MeshPart:
    """A mesh part with color information."""
    name: str
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)
    color: tuple[float, float, float] | None = None  # RGB 0-1


@dataclass
class ConversionResult:
    """Result of a conversion operation."""
    success: bool
    message: str
    output_path: Path | None = None
    vertex_count: int = 0
    face_count: int = 0


def read_step_file_xde(file_path: Path):
    """Read a STEP file using XDE to preserve colors and structure."""
    # Create document
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))

    # Read STEP file with color and name modes enabled
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    reader.SetLayerMode(True)

    status = reader.ReadFile(str(file_path))
    if status != IFSelect_RetDone:
        raise ValueError(f"Failed to read STEP file: {file_path}")

    reader.Transfer(doc)

    # Get tools
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

    return doc, shape_tool, color_tool


def get_shape_color(shape, color_tool) -> tuple[float, float, float] | None:
    """Get color for a shape."""
    color = Quantity_Color()

    # Try different color types
    for color_type in [XCAFDoc_ColorSurf, XCAFDoc_ColorGen, XCAFDoc_ColorCurv]:
        if color_tool.GetColor(shape, color_type, color):
            return (color.Red(), color.Green(), color.Blue())

    return None


def get_label_color(label, color_tool) -> tuple[float, float, float] | None:
    """Get color for a label by getting its shape first."""
    shape = XCAFDoc_ShapeTool.GetShape_s(label)
    if shape is None:
        return None
    return get_shape_color(shape, color_tool)


def extract_mesh_from_shape(shape, color: tuple[float, float, float] | None, name: str) -> MeshPart | None:
    """Extract mesh data from a single shape."""
    vertices = []
    faces = []
    normals = []
    current_index = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    while explorer.More():
        topo_face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(topo_face, location)

        if triangulation is not None:
            transform = location.Transformation()

            face_vertex_start = current_index
            for i in range(1, triangulation.NbNodes() + 1):
                node = triangulation.Node(i)
                transformed = node.Transformed(transform)
                vertex = (transformed.X(), transformed.Y(), transformed.Z())
                vertices.append(vertex)
                current_index += 1

            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                n1, n2, n3 = triangle.Get()

                v1 = vertices[face_vertex_start + n1 - 1]
                v2 = vertices[face_vertex_start + n2 - 1]
                v3 = vertices[face_vertex_start + n3 - 1]

                edge1 = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
                edge2 = (v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2])

                normal = (
                    edge1[1] * edge2[2] - edge1[2] * edge2[1],
                    edge1[2] * edge2[0] - edge1[0] * edge2[2],
                    edge1[0] * edge2[1] - edge1[1] * edge2[0]
                )

                length = (normal[0]**2 + normal[1]**2 + normal[2]**2) ** 0.5
                if length > 0:
                    normal = (normal[0]/length, normal[1]/length, normal[2]/length)
                else:
                    normal = (0.0, 0.0, 1.0)

                normals.append(normal)

                faces.append((
                    face_vertex_start + n1 - 1,
                    face_vertex_start + n2 - 1,
                    face_vertex_start + n3 - 1
                ))

        explorer.Next()

    if not vertices:
        return None

    return MeshPart(
        name=name,
        vertices=vertices,
        faces=faces,
        normals=normals,
        color=color
    )


def process_shape_recursive(shape, color_tool, shape_tool, parts: list[MeshPart], parent_color=None, name_prefix="Part"):
    """Recursively process shapes and extract meshes with colors."""
    # Get color for this shape
    color = get_shape_color(shape, color_tool)
    if color is None:
        color = parent_color

    shape_type = shape.ShapeType()

    if shape_type in [TopAbs_SOLID, TopAbs_SHELL]:
        # This is a solid/shell - extract mesh
        part = extract_mesh_from_shape(shape, color, f"{name_prefix}_{len(parts)}")
        if part:
            parts.append(part)
    elif shape_type == TopAbs_COMPOUND:
        # Compound - iterate children
        from OCP.TopoDS import TopoDS_Iterator
        it = TopoDS_Iterator(shape)
        idx = 0
        while it.More():
            child = it.Value()
            process_shape_recursive(child, color_tool, shape_tool, parts, color, f"{name_prefix}_{idx}")
            it.Next()
            idx += 1
    else:
        # Try to extract mesh anyway
        part = extract_mesh_from_shape(shape, color, f"{name_prefix}_{len(parts)}")
        if part:
            parts.append(part)


def extract_all_meshes(doc, shape_tool, color_tool, linear_deflection: float = 1.0, angular_deflection: float = 0.1) -> list[MeshPart]:
    """Extract all meshes with colors from the document."""
    parts = []

    # Get all free shapes (top-level shapes)
    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)

    print(f"  Found {labels.Length()} top-level shape(s)")

    for i in range(1, labels.Length() + 1):
        label = labels.Value(i)
        shape = XCAFDoc_ShapeTool.GetShape_s(label)

        if shape is None:
            continue

        # Tessellate the shape
        mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
        mesh.Perform()

        # Get color from label
        color = get_label_color(label, color_tool)

        # Process shape recursively
        process_shape_recursive(shape, color_tool, shape_tool, parts, color, f"Shape{i}")

    return parts


def color_to_material_name(color: tuple[float, float, float]) -> str:
    """Convert RGB (0-1) color to Fusion-style material name."""
    r = int(round(color[0] * 255))
    g = int(round(color[1] * 255))
    b = int(round(color[2] * 255))
    return f"Opaque({r},{g},{b})"


def write_obj_with_mtl(
    output_path: Path,
    parts: list[MeshPart]
) -> None:
    """Write mesh data to OBJ file with MTL material file (Fusion-compatible format).

    Fusion structure:
    1. All vertices (v) first
    2. All texture coordinates (vt) - vertex-based
    3. All normals (vn) - vertex-based
    4. Groups with usemtl and faces only
    """
    # MTL file name matches OBJ file name
    mtl_path = output_path.with_suffix(".mtl")
    mtl_name = mtl_path.name

    # Collect unique colors and assign material names (Fusion-style)
    color_to_material = {}
    for part in parts:
        if part.color and part.color not in color_to_material:
            color_to_material[part.color] = color_to_material_name(part.color)

    # Default material for parts without color
    default_material = "Opaque(192,192,192)"

    # Write MTL file (Fusion-compatible: only Kd)
    with open(mtl_path, 'w', encoding='utf-8') as f:
        f.write("# WaveFront *.mtl file (generated by STEP Converter)\n\n")

        # Default material
        f.write(f"newmtl {default_material}\n")
        f.write("Kd 0.752941 0.752941 0.752941\n\n")

        # Color materials
        for color, mat_name in color_to_material.items():
            f.write(f"newmtl {mat_name}\n")
            f.write(f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n\n")

    # Scale factor: mm to cm (Fusion uses cm)
    SCALE = 0.1

    # Build global lists and compute per-vertex normals
    all_vertices = []
    all_texcoords = []
    all_normals = []
    part_face_data = []

    for part in parts:
        start_v = len(all_vertices)
        start_vt = len(all_texcoords)
        start_vn = len(all_normals)

        # Add vertices (scaled to cm)
        for v in part.vertices:
            all_vertices.append((v[0] * SCALE, v[1] * SCALE, v[2] * SCALE))

        # Compute per-vertex normals by averaging face normals
        vertex_normals = {}  # vertex_index -> list of normals
        for i, face in enumerate(part.faces):
            normal = part.normals[i] if i < len(part.normals) else (0, 0, 1)
            for vi in face:
                if vi not in vertex_normals:
                    vertex_normals[vi] = []
                vertex_normals[vi].append(normal)

        # Average normals for each vertex
        averaged_normals = []
        for vi in range(len(part.vertices)):
            if vi in vertex_normals:
                normals = vertex_normals[vi]
                avg_x = sum(n[0] for n in normals) / len(normals)
                avg_y = sum(n[1] for n in normals) / len(normals)
                avg_z = sum(n[2] for n in normals) / len(normals)
                # Normalize
                length = (avg_x**2 + avg_y**2 + avg_z**2) ** 0.5
                if length > 0:
                    averaged_normals.append((avg_x/length, avg_y/length, avg_z/length))
                else:
                    averaged_normals.append((0, 0, 1))
            else:
                averaged_normals.append((0, 0, 1))

        # Add normals (one per vertex)
        for n in averaged_normals:
            all_normals.append(n)

        # Add texture coordinates (one per vertex, dummy values)
        for _ in range(len(part.vertices)):
            all_texcoords.append((0.0, 0.0))

        # Get material name
        if part.color and part.color in color_to_material:
            mat_name = color_to_material[part.color]
        else:
            mat_name = default_material

        # Store face data with global indices
        faces_global = []
        for face in part.faces:
            v1 = start_v + face[0] + 1
            v2 = start_v + face[1] + 1
            v3 = start_v + face[2] + 1
            vt1 = start_vt + face[0] + 1
            vt2 = start_vt + face[1] + 1
            vt3 = start_vt + face[2] + 1
            vn1 = start_vn + face[0] + 1
            vn2 = start_vn + face[1] + 1
            vn3 = start_vn + face[2] + 1
            faces_global.append((v1, vt1, vn1, v2, vt2, vn2, v3, vt3, vn3))

        part_face_data.append((part.name, faces_global, mat_name))

    # Write OBJ file (Fusion-compatible structure)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# WaveFront *.obj file (generated by STEP Converter)\n\n")
        f.write(f"mtllib {mtl_name}\n\n")

        # 1. Write all vertices first
        for v in all_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")

        # 2. Write texture coordinates (one per vertex)
        for vt in all_texcoords:
            f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
        f.write("\n")

        # 3. Write all normals (one per vertex)
        for n in all_normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("\n")

        # 4. Write groups with usemtl and faces
        for part_name, faces, mat_name in part_face_data:
            f.write(f"g {part_name}\n")
            f.write(f"usemtl {mat_name}\n")
            for face in faces:
                v1, vt1, vn1, v2, vt2, vn2, v3, vt3, vn3 = face
                f.write(f"f {v1}/{vt1}/{vn1} {v2}/{vt2}/{vn2} {v3}/{vt3}/{vn3}\n")
            f.write("\n")


def write_fbx_file(
    output_path: Path,
    parts: list[MeshPart],
    model_name: str = "Model"
) -> None:
    """Write mesh data to an FBX ASCII file with materials per part."""
    # Collect unique colors for materials
    unique_colors = {}
    for part in parts:
        if part.color and part.color not in unique_colors:
            unique_colors[part.color] = len(unique_colors)

    # Default gray color if no colors
    default_color = (0.75, 0.75, 0.75)
    if default_color not in unique_colors:
        unique_colors[default_color] = len(unique_colors)

    # ID bases
    GEOMETRY_BASE = 2000000000
    MODEL_BASE = 3000000000
    MATERIAL_BASE = 4000000000

    num_parts = len(parts)
    num_materials = len(unique_colors)
    total_objects = 1 + num_parts * 2 + num_materials  # GlobalSettings + Geometry*n + Model*n + Material*n

    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("; FBX 7.4.0 project file\n")
        f.write("; Generated by STEP Converter\n")
        f.write(";\n\n")

        f.write("FBXHeaderExtension:  {\n")
        f.write("\tFBXHeaderVersion: 1003\n")
        f.write("\tFBXVersion: 7400\n")
        f.write("\tCreationTimeStamp:  {\n")
        f.write("\t\tVersion: 1000\n")
        f.write("\t\tYear: 2024\n")
        f.write("\t\tMonth: 1\n")
        f.write("\t\tDay: 1\n")
        f.write("\t\tHour: 0\n")
        f.write("\t\tMinute: 0\n")
        f.write("\t\tSecond: 0\n")
        f.write("\t\tMillisecond: 0\n")
        f.write("\t}\n")
        f.write('\tCreator: "STEP Converter"\n')
        f.write("}\n\n")

        # Global Settings
        f.write("GlobalSettings:  {\n")
        f.write("\tVersion: 1000\n")
        f.write("\tProperties70:  {\n")
        f.write('\t\tP: "UpAxis", "int", "Integer", "",1\n')
        f.write('\t\tP: "UpAxisSign", "int", "Integer", "",1\n')
        f.write('\t\tP: "FrontAxis", "int", "Integer", "",2\n')
        f.write('\t\tP: "FrontAxisSign", "int", "Integer", "",1\n')
        f.write('\t\tP: "CoordAxis", "int", "Integer", "",0\n')
        f.write('\t\tP: "CoordAxisSign", "int", "Integer", "",1\n')
        f.write('\t\tP: "OriginalUpAxis", "int", "Integer", "",-1\n')
        f.write('\t\tP: "OriginalUpAxisSign", "int", "Integer", "",1\n')
        f.write('\t\tP: "UnitScaleFactor", "double", "Number", "",1\n')
        f.write("\t}\n")
        f.write("}\n\n")

        # Documents
        f.write("Documents:  {\n")
        f.write("\tCount: 1\n")
        f.write('\tDocument: 1000000000, "", "Scene" {\n')
        f.write("\t\tProperties70:  {\n")
        f.write('\t\t\tP: "SourceObject", "object", "", ""\n')
        f.write('\t\t\tP: "ActiveAnimStackName", "KString", "", "", ""\n')
        f.write("\t\t}\n")
        f.write("\t\tRootNode: 0\n")
        f.write("\t}\n")
        f.write("}\n\n")

        f.write("References:  {\n")
        f.write("}\n\n")

        # Definitions
        f.write("Definitions:  {\n")
        f.write("\tVersion: 100\n")
        f.write(f"\tCount: {total_objects}\n")
        f.write('\tObjectType: "GlobalSettings" {\n')
        f.write("\t\tCount: 1\n")
        f.write("\t}\n")
        f.write('\tObjectType: "Geometry" {\n')
        f.write(f"\t\tCount: {num_parts}\n")
        f.write("\t}\n")
        f.write('\tObjectType: "Model" {\n')
        f.write(f"\t\tCount: {num_parts}\n")
        f.write("\t}\n")
        f.write('\tObjectType: "Material" {\n')
        f.write(f"\t\tCount: {num_materials}\n")
        f.write("\t}\n")
        f.write("}\n\n")

        # Objects
        f.write("Objects:  {\n")

        # Write each part as separate Geometry
        for part_idx, part in enumerate(parts):
            geom_id = GEOMETRY_BASE + part_idx
            part_name = part.name.replace(" ", "_")

            # Flatten vertices
            vertices_flat = []
            for v in part.vertices:
                vertices_flat.extend([v[0], v[1], v[2]])

            # Build polygon indices
            polygon_indices = []
            for face in part.faces:
                polygon_indices.append(face[0])
                polygon_indices.append(face[1])
                polygon_indices.append(face[2] ^ -1)  # Last vertex negated and -1

            # Flatten normals
            normals_flat = []
            for n in part.normals:
                normals_flat.extend([n[0], n[1], n[2]])

            f.write(f'\tGeometry: {geom_id}, "Geometry::{part_name}", "Mesh" {{\n')

            f.write(f"\t\tVertices: *{len(vertices_flat)} {{\n")
            f.write("\t\t\ta: ")
            f.write(",".join(f"{v:.6f}" for v in vertices_flat))
            f.write("\n\t\t}\n")

            f.write(f"\t\tPolygonVertexIndex: *{len(polygon_indices)} {{\n")
            f.write("\t\t\ta: ")
            f.write(",".join(str(i) for i in polygon_indices))
            f.write("\n\t\t}\n")

            f.write('\t\tLayerElementNormal: 0 {\n')
            f.write('\t\t\tVersion: 102\n')
            f.write('\t\t\tName: "Normals"\n')
            f.write('\t\t\tMappingInformationType: "ByPolygon"\n')
            f.write('\t\t\tReferenceInformationType: "Direct"\n')
            f.write(f"\t\t\tNormals: *{len(normals_flat)} {{\n")
            f.write("\t\t\t\ta: ")
            f.write(",".join(f"{n:.6f}" for n in normals_flat))
            f.write("\n\t\t\t}\n")
            f.write('\t\t}\n')

            f.write('\t\tLayer: 0 {\n')
            f.write('\t\t\tVersion: 100\n')
            f.write('\t\t\tLayerElement:  {\n')
            f.write('\t\t\t\tType: "LayerElementNormal"\n')
            f.write('\t\t\t\tTypedIndex: 0\n')
            f.write('\t\t\t}\n')
            f.write('\t\t}\n')

            f.write("\t}\n")

        # Write each part as separate Model
        for part_idx, part in enumerate(parts):
            model_id = MODEL_BASE + part_idx
            part_name = part.name.replace(" ", "_")

            f.write(f'\tModel: {model_id}, "Model::{part_name}", "Mesh" {{\n')
            f.write('\t\tVersion: 232\n')
            f.write('\t\tProperties70:  {\n')
            f.write('\t\t\tP: "ScalingMax", "Vector3D", "Vector", "",0,0,0\n')
            f.write('\t\t}\n')
            f.write('\t\tShading: T\n')
            f.write('\t\tCulling: "CullingOff"\n')
            f.write("\t}\n")

        # Write Materials
        for color, mat_idx in unique_colors.items():
            mat_id = MATERIAL_BASE + mat_idx
            r, g, b = color
            mat_name = f"Material_{int(r*255)}_{int(g*255)}_{int(b*255)}"

            f.write(f'\tMaterial: {mat_id}, "Material::{mat_name}", "" {{\n')
            f.write('\t\tVersion: 102\n')
            f.write('\t\tShadingModel: "lambert"\n')
            f.write('\t\tMultiLayer: 0\n')
            f.write('\t\tProperties70:  {\n')
            f.write(f'\t\t\tP: "DiffuseColor", "Color", "", "A",{r:.6f},{g:.6f},{b:.6f}\n')
            f.write(f'\t\t\tP: "Diffuse", "Vector3D", "Vector", "",{r:.6f},{g:.6f},{b:.6f}\n')
            f.write('\t\t}\n')
            f.write("\t}\n")

        f.write("}\n\n")

        # Connections
        f.write("Connections:  {\n")

        for part_idx, part in enumerate(parts):
            geom_id = GEOMETRY_BASE + part_idx
            model_id = MODEL_BASE + part_idx

            # Model -> Root
            f.write(f'\tC: "OO",{model_id},0\n')
            # Geometry -> Model
            f.write(f'\tC: "OO",{geom_id},{model_id}\n')

            # Material -> Model
            color = part.color if part.color else default_color
            mat_idx = unique_colors[color]
            mat_id = MATERIAL_BASE + mat_idx
            f.write(f'\tC: "OO",{mat_id},{model_id}\n')

        f.write("}\n")


def convert_step(
    input_path: Path,
    output_path: Path,
    output_format: OutputFormat = OutputFormat.OBJ,
    linear_deflection: float = 1.0,
    angular_deflection: float = 0.1
) -> ConversionResult:
    """Convert a STEP file to the specified format."""
    try:
        print(f"[1/4] Reading STEP file: {input_path}")
        doc, shape_tool, color_tool = read_step_file_xde(input_path)
        print("[1/4] Done reading STEP file")

        print("[2/4] Extracting meshes with colors...")
        parts = extract_all_meshes(doc, shape_tool, color_tool, linear_deflection, angular_deflection)
        print(f"[2/4] Done: {len(parts)} part(s) extracted")

        total_vertices = sum(len(p.vertices) for p in parts)
        total_faces = sum(len(p.faces) for p in parts)
        colors_found = sum(1 for p in parts if p.color is not None)

        print(f"[3/4] Total: {total_vertices} vertices, {total_faces} triangles, {colors_found} colored parts")

        if not parts:
            return ConversionResult(
                success=False,
                message="No geometry found in STEP file"
            )

        print(f"[4/4] Writing {output_format.value.upper()} file: {output_path}")

        if output_format == OutputFormat.OBJ:
            write_obj_with_mtl(output_path, parts)
        elif output_format == OutputFormat.FBX:
            model_name = input_path.stem
            write_fbx_file(output_path, parts, model_name)

        print(f"[4/4] Done writing {output_format.value.upper()} file")

        return ConversionResult(
            success=True,
            message="Conversion completed successfully",
            output_path=output_path,
            vertex_count=total_vertices,
            face_count=total_faces
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return ConversionResult(
            success=False,
            message=f"Conversion failed: {str(e)}"
        )


# Backward compatibility
def convert_step_to_obj(
    input_path: Path,
    output_path: Path,
    linear_deflection: float = 1.0,
    angular_deflection: float = 0.1
) -> ConversionResult:
    """Convert a STEP file to OBJ format (legacy function)."""
    return convert_step(input_path, output_path, OutputFormat.OBJ, linear_deflection, angular_deflection)
