// Gmsh project created on Fri Apr 05 21:01:20 2024
SetFactory("OpenCASCADE");
mesh_size0=0.0044 * 100;
scale=1;
cx = 2*Pi*scale;
cy = 2*Pi*scale;

side_length=0.8*Pi * scale;
h1 = ((Sqrt(3)/2)/3)* side_length;
h2 = 2 * h1;

Point(1) = {Pi * scale, Pi * scale, 0, mesh_size0};
Point(2) = {Pi * scale, 3*Pi*scale, 0, mesh_size0};
Point(3) = {3*Pi*scale, 3*Pi*scale, 0, mesh_size0};
Point(4) = {3*Pi*scale, Pi * scale, 0, mesh_size0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// https://math.stackexchange.com/questions/2778306/computing-the-coordinate-of-an-equilateral-triangle
Point(5) = {cx-side_length/2, cy-h1, 0, mesh_size0};
Point(6) = {cx+side_length/2, cy-h1, 0, mesh_size0};
Point(7) = {cx, cy+h2, 0, mesh_size0};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 5};

Rotate {{0, 0, 1}, {2*Pi*scale, 2*Pi*scale, 0}, Pi/2} {
  Curve{7}; Curve{6}; Curve{5}; 
}

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {7, 5, 6};
Plane Surface(1) = {1, 2};

Extrude {0, 0, 0.005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}

//+
Physical Surface("inlet", 22) = {2};
Physical Surface("outlet", 23) = {4};
Physical Surface("obstacle", 24) = {8, 6, 7};
Physical Surface("wall", 25) = {3, 5};
Physical Surface("frontAndBack", 26) = {9, 1};
Physical Volume("volume", 27) = {1};
