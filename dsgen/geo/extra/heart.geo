// Gmsh project created on Fri Apr 05 21:01:20 2024
SetFactory("OpenCASCADE");
mesh_size0=0.005 * 10;
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



Point(5) = {cx, cy-Ri, 0, mesh_size0};
Point(7) = {cx+Ri, cy-side_length/2, 0, mesh_size0};
Point(8) = {cx+Ri, cy+side_length/2, 0, mesh_size0};
Point(9) = {cx+side_length/2, cy+Ri, 0, mesh_size0};
Point(13) = {cx, cy+side_length/2, 0, mesh_size0};

Point(10) = {cx-side_length/2, cy+Ri, 0, mesh_size0};
Point(11) = {cx-Ri, cy+side_length/2, 0, mesh_size0};
Point(12) = {cx-Ri, cy-side_length/2, 0, mesh_size0};

Bezier(14) = {13, 9, 8, 7, 5};
Bezier(15) = {13, 10, 11, 12, 5};

//+
//+
Rotate {{0, 0, 1}, {2*Pi*scale, 2*Pi*scale, 0}, Pi/2} {
  Curve{15}; Curve{14}; 
}
//+
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {14, -15};
Plane Surface(1) = {1, 2};

Extrude {0, 0, 0.0005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}//+//+

Physical Surface("inlet", 28) = {2};
Physical Surface("outlet", 29) = {4};
Physical Surface("wall", 30) = {3, 5};
Physical Surface("obstacle", 31) = {6, 7};
Physical Surface("frontAndBack", 32) = {8, 1};
Physical Volume("volume", 33) = {1};
