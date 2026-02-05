"""STEP to OBJ/FBX conversion logic using Open Cascade XDE."""

from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import time

from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_SHELL, TopAbs_COMPOUND, TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS, TopoDS_Iterator
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool, XCAFDoc_ColorTool, XCAFDoc_ColorSurf, XCAFDoc_ColorGen, XCAFDoc_ColorCurv, XCAFDoc_Color
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCP.GeomLProp import GeomLProp_SLProps
from OCP.gp import gp_Pnt2d


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
    """Get color for a label directly."""
    color = Quantity_Color()

    # Try different color types on the LABEL directly
    # Use static method GetColor_s which takes Label (instance method only takes Shape)
    for color_type in [XCAFDoc_ColorSurf, XCAFDoc_ColorGen, XCAFDoc_ColorCurv]:
        # signature: static GetColor_s(L: Label, type: Type, C: Color) -> bool
        if XCAFDoc_ColorTool.GetColor_s(label, color_type, color):
            return (color.Red(), color.Green(), color.Blue())
            
    return None


def extract_meshes_by_color(shape, color_tool, parent_color: tuple[float, float, float] | None, name_prefix: str) -> list[MeshPart]:
    """Extract mesh data from a shape, splitting by face colors, with vertex deduplication."""
    # Map color -> {vertices, faces, normals, unique_map}
    parts_data = {}

    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    while explorer.More():
        topo_face = TopoDS.Face_s(explorer.Current())
        
        # Get face color or fallback to parent
        face_color = get_shape_color(topo_face, color_tool)
        if face_color is None:
            face_color = parent_color
            
        # Use a key for the dictionary (tuple is hashable)
        color_key = face_color
        
        if color_key not in parts_data:
            parts_data[color_key] = {
                "vertices": [],
                "faces": [],
                "normals": [],
                "unique_map": {}  # (x,y,z, nx,ny,nz) -> index
            }
            
        data = parts_data[color_key]
        
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(topo_face, location)

        if triangulation is not None:
            # Prepare properties for smooth normals
            surface = BRep_Tool.Surface_s(topo_face)
            has_uv = triangulation.HasUVNodes()
            face_reversed = topo_face.Orientation() == TopAbs_REVERSED

            transform = location.Transformation()
            
            # Temporary map for this face to map node_index -> global_index
            # This is needed because triangles invoke nodes by 1-based index local to the triangulation
            local_node_map = {} 
            
            # Process vertices
            for i in range(1, triangulation.NbNodes() + 1):
                node = triangulation.Node(i)
                transformed = node.Transformed(transform)
                vx, vy, vz = transformed.X(), transformed.Y(), transformed.Z()
                
                # Calculate smooth normal if possible, else default up
                nx, ny, nz = 0.0, 0.0, 1.0
                
                if has_uv and surface:
                    uv = triangulation.UVNode(i)
                    props = GeomLProp_SLProps(surface, uv.X(), uv.Y(), 1, 1e-5)
                    if props.IsNormalDefined():
                        n = props.Normal()
                        # Apply transformation
                        n_transformed = n.Transformed(transform) 
                        nx, ny, nz = n_transformed.X(), n_transformed.Y(), n_transformed.Z()
                        
                        # Flip if face is reversed
                        if face_reversed:
                            nx, ny, nz = -nx, -ny, -nz
                            
                        # Normalize
                        length = (nx*nx + ny*ny + nz*nz) ** 0.5
                        if length > 1e-9:
                            nx, ny, nz = nx/length, ny/length, nz/length

                # Deduplication Key: Position + Normal
                # Using 6 decimal places for deduplication tolerance
                key = (
                    round(vx, 6), round(vy, 6), round(vz, 6),
                    round(nx, 6), round(ny, 6), round(nz, 6)
                )

                if key in data["unique_map"]:
                    global_idx = data["unique_map"][key]
                else:
                    global_idx = len(data["vertices"])
                    data["vertices"].append((vx, vy, vz))
                    data["normals"].append((nx, ny, nz))
                    data["unique_map"][key] = global_idx
                
                local_node_map[i] = global_idx

            # Add triangles
            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                n1, n2, n3 = triangle.Get()

                v1_idx = local_node_map[n1]
                v2_idx = local_node_map[n2]
                v3_idx = local_node_map[n3]

                # Reverse winding order if face is reversed
                if face_reversed:
                    data["faces"].append((v1_idx, v3_idx, v2_idx))
                else:
                    data["faces"].append((v1_idx, v2_idx, v3_idx))

        explorer.Next()

    result_parts = []
    for i, (color, data) in enumerate(parts_data.items()):
        if not data["vertices"]:
            continue
            
        part_name = f"{name_prefix}_{i}" if len(parts_data) > 1 else name_prefix
        
        result_parts.append(MeshPart(
            name=part_name,
            vertices=data["vertices"],
            faces=data["faces"],
            normals=data["normals"],
            color=color
        ))

    return result_parts


def process_label_recursive(label, color_tool, shape_tool, parts: list[MeshPart], parent_color=None, parent_loc=None, name_prefix="Part"):
    """Recursively process XCAF labels to handle Assembly colors correctly."""
    # DEBUG: Verify strict static method usage
    # print(f"DEBUG: Processing label {name_prefix}")
    
    if parent_loc is None:
        parent_loc = TopLoc_Location()

    # 1. Resolve Color
    # Use label color if present, else inherit parent
    color = get_label_color(label, color_tool)
    if color is None:
        color = parent_color

    # 2. Resolve Location and Target
    # Get location of this node (relative to parent)
    local_loc = XCAFDoc_ShapeTool.GetLocation_s(label)
    
    # Calculate accumulated location for passing to children
    # Note: OCP Location multiplication is usually parent * child
    new_parent_loc = parent_loc * local_loc

    # Resolve reference (Component -> Part/Assembly)
    target_label = TDF_Label()
    has_ref = XCAFDoc_ShapeTool.GetReferredShape_s(label, target_label)
    if not has_ref or target_label.IsNull():
        target_label = label

    # 2. Resolve Color (Fallback to Prototype/Target)
    if color is None:
        # Check if the referred shape (prototype) has a color
        if not target_label.IsEqual(label):
             color = get_label_color(target_label, color_tool)

    # 3. Inherit Parent Color
    if color is None:
        color = parent_color

    # Check if Assembly
    if XCAFDoc_ShapeTool.IsAssembly_s(target_label):
        # Iterate components
        comps = TDF_LabelSequence()
        XCAFDoc_ShapeTool.GetComponents_s(target_label, comps)
        
        for i in range(1, comps.Length() + 1):
            child_label = comps.Value(i)
            process_label_recursive(
                child_label, 
                color_tool, 
                shape_tool, 
                parts, 
                color, 
                new_parent_loc, 
                f"{name_prefix}_{i}"
            )
    else:
        # Leaf (Simple Shape)
        # Get the shape from the label (this includes local_loc)
        shape = XCAFDoc_ShapeTool.GetShape_s(label)
        
        if shape is not None:
            # Apply the accumulated parent location
            # shape has 'local_loc'. we need 'parent_loc * local_loc'.
            # shape.Moved(parent_loc) applies parent_loc on top.
            final_shape = shape.Moved(parent_loc)
            
            # Extract mesh (supports multiple colors within the shape if any)
            new_parts = extract_meshes_by_color(final_shape, color_tool, color, f"{name_prefix}_{len(parts)}")
            parts.extend(new_parts)


def extract_all_meshes(doc, shape_tool, color_tool, linear_deflection: float = 0.01, angular_deflection: float = 0.1) -> list[MeshPart]:
    """Extract all meshes with colors from the document."""
    parts = []

    # Get all free shapes (top-level shapes)
    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)

    print(f"  Found {labels.Length()} top-level shape(s)")

    for i in range(1, labels.Length() + 1):
        label = labels.Value(i)
        
        # 1. Mesh the geometry first (using the Shape)
        # This ensures all sub-shapes have triangulation
        shape = XCAFDoc_ShapeTool.GetShape_s(label)
        if shape is None:
            continue
            
        # Tessellate the shape
        mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
        mesh.Perform()
        
        # 2. Traverse attributes (Color, Structure) using Labels
        # Reset location for root
        identity_loc = TopLoc_Location()
        process_label_recursive(label, color_tool, shape_tool, parts, None, identity_loc, f"Shape{i}")

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

        # Normals are already per-vertex from extract_meshes_by_color
        # Just use them directly
        for n in part.normals:
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
    # Brighter default
    default_color = (0.85, 0.85, 0.85)
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

            # --- Optimization: Positional Deduplication for Vertices ---
            unique_pos_map = {} # (x,y,z) -> new_index
            new_vertices = []
            
            # Map old_vertex_index -> new_vertex_index
            old_to_new_v_map = [0] * len(part.vertices)
            
            for i, v in enumerate(part.vertices):
                # Use 6 decimals for positional tolerance
                key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
                if key in unique_pos_map:
                    old_to_new_v_map[i] = unique_pos_map[key]
                else:
                    idx = len(new_vertices)
                    new_vertices.append(v)
                    unique_pos_map[key] = idx
                    old_to_new_v_map[i] = idx
            
            # Flatten new vertices
            vertices_flat = []
            for v in new_vertices:
                vertices_flat.extend([v[0], v[1], v[2]])

            # Build polygon indices utilizing the new vertex mapping
            polygon_indices = []
            for face in part.faces:
                v1 = old_to_new_v_map[face[0]]
                v2 = old_to_new_v_map[face[1]]
                v3 = old_to_new_v_map[face[2]]
                
                polygon_indices.append(v1)
                polygon_indices.append(v2)
                polygon_indices.append(v3 ^ -1)  # Last vertex negated and -1

            # --- Optimization: Indexed Normals ---
            # Gather unique key normals to compress data
            unique_normal_map = {} # (nx,ny,nz) -> index
            unique_normals_list = []
            normal_indices_flat = []
            
            for face in part.faces:
                for v_idx in face:
                    n = part.normals[v_idx]
                    key = (round(n[0], 6), round(n[1], 6), round(n[2], 6))
                    
                    if key in unique_normal_map:
                        idx = unique_normal_map[key]
                    else:
                        idx = len(unique_normals_list)
                        unique_normals_list.append(n)
                        unique_normal_map[key] = idx
                    
                    normal_indices_flat.append(idx)
            
            # Flatten unique normals
            normals_flat_values = []
            for n in unique_normals_list:
                normals_flat_values.extend([n[0], n[1], n[2]])


            f.write(f'\tGeometry: {geom_id}, "Geometry::{part_name}", "Mesh" {{\n')

            f.write(f"\t\tVertices: *{len(vertices_flat)} {{\n")
            f.write("\t\t\ta: ")
            f.write(",".join(f"{v:.4f}" for v in vertices_flat))
            f.write("\n\t\t}\n")

            f.write(f"\t\tPolygonVertexIndex: *{len(polygon_indices)} {{\n")
            f.write("\t\t\ta: ")
            f.write(",".join(str(i) for i in polygon_indices))
            f.write("\n\t\t}\n")

            f.write(f'\t\tLayerElementNormal: 0 {{\n')
            f.write('\t\t\tVersion: 102\n')
            f.write('\t\t\tName: "Normals"\n')
            f.write('\t\t\tMappingInformationType: "ByPolygonVertex"\n')
            f.write('\t\t\tReferenceInformationType: "IndexToDirect"\n') 
            
            f.write(f"\t\t\tNormals: *{len(normals_flat_values)} {{\n")
            f.write("\t\t\t\ta: ")
            f.write(",".join(f"{n:.4f}" for n in normals_flat_values))
            f.write("\n\t\t\t}\n")
            
            f.write(f"\t\t\tNormalsIndex: *{len(normal_indices_flat)} {{\n")
            f.write("\t\t\t\ta: ")
            f.write(",".join(str(i) for i in normal_indices_flat))
            f.write("\n\t\t\t}\n")
            
            f.write('\t\t}\n')

            f.write(f'\t\tLayerElementMaterial: 0 {{\n')
            f.write('\t\t\tVersion: 101\n')
            f.write('\t\t\tName: "Materials"\n')
            f.write('\t\t\tMappingInformationType: "AllSame"\n')
            f.write('\t\t\tReferenceInformationType: "IndexToDirect"\n')
            f.write('\t\t\tMaterials: *1 {\n')
            f.write('\t\t\t\ta: 0\n')
            f.write('\t\t\t}\n')
            f.write('\t\t}\n')

            f.write('\t\tLayer: 0 {\n')
            f.write('\t\t\tVersion: 100\n')
            f.write('\t\t\tLayerElement:  {\n')
            f.write('\t\t\t\tType: "LayerElementNormal"\n')
            f.write('\t\t\t\tTypedIndex: 0\n')
            f.write('\t\t\t}\n')
            f.write('\t\t\tLayerElement:  {\n')
            f.write('\t\t\t\tType: "LayerElementMaterial"\n')
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
            f.write('\t\tShadingModel: "phong"\n')
            f.write('\t\tMultiLayer: 0\n')
            f.write('\t\tProperties70:  {\n')
            f.write(f'\t\t\tP: "DiffuseColor", "Color", "", "A",{r:.4f},{g:.4f},{b:.4f}\n')
            f.write(f'\t\t\tP: "Diffuse", "Vector3D", "Vector", "",{r:.4f},{g:.4f},{b:.4f}\n')
            # Add Ambient/Specular for brighter look
            f.write(f'\t\t\tP: "AmbientColor", "Color", "", "A",0.5,0.5,0.5\n')
            f.write(f'\t\t\tP: "SpecularColor", "Color", "", "A",0.4,0.4,0.4\n')
            f.write(f'\t\t\tP: "Shininess", "double", "Number", "",16.0\n')
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
