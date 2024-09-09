// Gmsh project created on Fri Apr 05 21:01:20 2024
SetFactory("OpenCASCADE");
mesh_size0=0.005 * 100;
scale=1;
cx = 2*Pi*scale;
cy = 2*Pi*scale;
a = 0.8 * Pi * scale;
b = 0.16 * Pi * scale;

Point(1) = {Pi * scale, Pi * scale, 0, mesh_size0};
Point(2) = {Pi * scale, 3*Pi*scale, 0, mesh_size0};
Point(3) = {3*Pi*scale, 3*Pi*scale, 0, mesh_size0};
Point(4) = {3*Pi*scale, Pi * scale, 0, mesh_size0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};



Point(5) = {cx-a/2 + b, cy - a/2, 0, mesh_size0};
Point(6) = {cx+a/2 - b, cy - a/2, 0, mesh_size0};
Point(7) = {cx+a/2 -b, cy - a/2+b, 0, mesh_size0};
Point(8) = {cx+a/2, cy - a/2+b, 0, mesh_size0};
Point(9) = {cx+a/2, cy + a/2-b, 0, mesh_size0};
Point(10) = {cx+a/2-b, cy + a/2-b, 0, mesh_size0};
Point(11) = {cx+a/2 -b, cy + a/2, 0, mesh_size0};
Point(12) = {cx-a/2 + b, cy + a/2, 0, mesh_size0};
Point(13) = {cx-a/2 + b, cy + a/2 -b, 0, mesh_size0};
Point(14) = {cx-a/2 , cy + a/2 -b, 0, mesh_size0};
Point(15) = {cx-a/2 , cy - a/2 +b, 0, mesh_size0};
Point(16) = {cx-a/2 + b , cy - a/2 +b, 0, mesh_size0};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {11, 10};
Line(11) = {12, 11};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 16};
Line(16) = {16, 5};
Rotate {{0, 0, 1}, {2*Pi*scale, 2*Pi*scale, 0}, Pi/4} {
  Curve{14}; Curve{13}; Curve{12}; Curve{11}; Curve{10}; Curve{9}; Curve{8}; Curve{7}; Curve{6}; Curve{5}; Curve{16}; Curve{15}; 
}

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {11, 10, -9, -8, -7, -6, -5, -16, -15, -14, -13, -12};
Plane Surface(1) = {1, 2};

Extrude {0, 0, 0.005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}

//+
Physical Surface("inlet", 49) = {2};
Physical Surface("outlet", 50) = {4};
Physical Surface("obstacle", 51) = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
Physical Surface("wall", 52) = {5, 3};
Physical Surface("frontAndBack", 53) = {18, 1};
Physical Volume("volume", 54) = {1};
