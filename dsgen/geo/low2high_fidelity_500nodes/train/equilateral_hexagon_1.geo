// Gmsh project created on Fri Apr 05 21:01:20 2024
SetFactory("OpenCASCADE");
mesh_size0=0.0048 * 100;
scale=1;
cx = 2*Pi*scale;
cy = 2*Pi*scale;
side_length=0.8*Pi * (Sqrt(3)/3)* scale;


Point(1) = {Pi * scale, Pi * scale, 0, mesh_size0};
Point(2) = {Pi * scale, 3*Pi*scale, 0, mesh_size0};
Point(3) = {3*Pi*scale, 3*Pi*scale, 0, mesh_size0};
Point(4) = {3*Pi*scale, Pi * scale, 0, mesh_size0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


// https://www.quora.com/Is-there-a-simple-way-to-draw-a-hexagon-with-a-known-width-across-the-flat-sides-instead-of-corner-to-corner
a = side_length/(Sqrt(3)/3)/2  ;
r =  a / (Sqrt(3)/2);

Point(5) = {cx-side_length/2, cy-a, 0, mesh_size0};
Point(6) = {cx+side_length/2, cy-a, 0, mesh_size0};
Point(7) = {cx+r, cy, 0, mesh_size0};
Point(8) = {cx+side_length/2, cy+a, 0, mesh_size0};
Point(9) = {cx-side_length/2, cy+a, 0, mesh_size0};
Point(10) = {cx-r, cy, 0, mesh_size0};

Line(5) = {10, 5};
Line(6) = {5, 6};
Line(7) = {6, 7};
Line(8) = {7, 8};
Line(9) = {8, 9};
Line(10) = {9, 10};

Rotate {{0, 0, 1}, {2*Pi*scale, 2*Pi*scale, 0}, 0} {
  Curve{10}; Curve{9}; Curve{8}; Curve{7}; Curve{5}; Curve{6}; 
}

Curve Loop(1) = {10, 5, 6, 7, 8, 9};
Curve Loop(2) = {1, 2, 3, 4};
Plane Surface(1) = {1, 2};

Extrude {0, 0, 0.005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}
//+
Physical Surface("inlet", 31) = {8};
Physical Surface("outlet", 32) = {10};
Physical Surface("obstacle", 33) = {6, 5, 4, 3, 7, 2};
Physical Surface("wall", 34) = {9, 11};
Physical Surface("frontAndBack", 35) = {12, 1};
Physical Volume("volume", 36) = {1};
