// Gmsh project created on Fri Apr 05 21:01:20 2024
SetFactory("OpenCASCADE");
mesh_size0=0.0050 * 100;
scale=1;
cx = 2*Pi*scale;
cy = 2*Pi*scale;
side_length=0.8*Pi * (Sqrt(2) -1)* scale;

h=2.414*side_length;
Ri = h/2;


Point(1) = {Pi * scale, Pi * scale, 0, mesh_size0};
Point(2) = {Pi * scale, 3*Pi*scale, 0, mesh_size0};
Point(3) = {3*Pi*scale, 3*Pi*scale, 0, mesh_size0};
Point(4) = {3*Pi*scale, Pi * scale, 0, mesh_size0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};



Point(5) = {cx-side_length/2, cy-Ri, 0, mesh_size0};
Point(6) = {cx+side_length/2, cy-Ri, 0, mesh_size0};
Point(7) = {cx+Ri, cy-side_length/2, 0, mesh_size0};
Point(8) = {cx+Ri, cy+side_length/2, 0, mesh_size0};
Point(9) = {cx+side_length/2, cy+Ri, 0, mesh_size0};
Point(10) = {cx-side_length/2, cy+Ri, 0, mesh_size0};
Point(11) = {cx-Ri, cy+side_length/2, 0, mesh_size0};
Point(12) = {cx-Ri, cy-side_length/2, 0, mesh_size0};

Line(5) = {11, 12};
Line(6) = {12, 5};
Line(7) = {5, 6};
Line(8) = {6, 7};
Line(9) = {7, 8};
Line(10) = {8, 9};
Line(11) = {9, 10};
Line(12) = {10, 11};
Rotate {{0, 0, 1}, {2*Pi*scale, 2*Pi*scale, 0}, 0} {
  Curve{12}; Curve{5}; Curve{6}; Curve{7}; Curve{8}; Curve{9}; Curve{11}; Curve{10}; 
}

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {12, 5, 6, 7, 8, 9, 10, 11};
Plane Surface(1) = {1, 2};

Extrude {0, 0, 0.005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}//+
Physical Surface("inlet", 37) = {2};
Physical Surface("outlet", 38) = {4};
Physical Surface("obstacle", 39) = {6, 13, 12, 11, 10, 9, 8, 7};
Physical Surface("wall", 40) = {3, 5};
Physical Surface("frontAndBack", 41) = {1, 14};
Physical Volume("volume", 42) = {1};
