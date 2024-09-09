// Gmsh project created on Fri Apr 05 21:01:20 2024
SetFactory("OpenCASCADE");
mesh_size0=0.005 * 10;
scale=1;
cx = 2*Pi*scale;
cy = 2*Pi*scale;
side_length=0.8*Pi * (Sqrt(2) -1)* scale;

h=2.414*side_length;
Ri = h/2;
side_length=Ri* scale;


Point(1) = {Pi * scale, Pi * scale, 0, mesh_size0};
Point(2) = {Pi * scale, 3*Pi*scale, 0, mesh_size0};
Point(3) = {3*Pi*scale, 3*Pi*scale, 0, mesh_size0};
Point(4) = {3*Pi*scale, Pi * scale, 0, mesh_size0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};



Point(5) = {cx, cy-side_length/4, 0, mesh_size0};
Point(7) = {cx+Ri/2, cy-side_length/2, 0, mesh_size0};
Point(8) = {cx+Ri/4, cy+side_length/2, 0, mesh_size0};
Point(9) = {cx+3*Ri/4, cy+side_length/2, 0, mesh_size0};
// Point(9) = {cx+Ri, cy+side_length/2, 0, mesh_size0};
Point(14) = {cx+Ri/3, cy+side_length/8, 0, mesh_size0};
Point(10) = {cx, cy+Ri, 0, mesh_size0};



Point(11) = {cx-Ri/4, cy+side_length/2, 0, mesh_size0};
Point(12) = {cx-3*Ri/4, cy+side_length/2, 0, mesh_size0};
Point(13) = {cx-Ri/3, cy+side_length/8, 0, mesh_size0};
Point(15) = {cx-Ri/2, cy-side_length/2, 0, mesh_size0};


//+
Line(5) = {10, 8};
//+
Line(6) = {9, 8};
//+
Line(7) = {9, 14};
//+
Line(8) = {14, 7};
//+
Line(9) = {5, 7};
//+
Line(10) = {5, 15};
//+
Line(11) = {15, 13};
//+
Line(12) = {13, 12};
//+
Line(13) = {11, 12};
//+
Line(14) = {11, 10};
//+
Translate {0,-side_length/8 , 0} {
  Curve{14}; Curve{13}; Curve{12}; Curve{11}; Curve{10}; Curve{9}; Curve{8}; Curve{7}; Curve{6}; Curve{5}; 
}
//+
Curve Loop(1) = {2, 3, 4, 1};
Curve Loop(2) = {6, -5, -14, 13, -12, -11, -10, 9, -8, -7};
Plane Surface(1) = {1, 2};
Extrude {0, 0, 0.0005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}//+//+
Physical Surface("inlet", 43) = {5};
Physical Surface("outlet", 44) = {3};
Physical Surface("wall", 45) = {4, 2};
Physical Surface("obstacle", 46) = {8, 7, 6, 15, 14, 13, 12, 11, 9, 10};
Physical Surface("frontAndBack", 47) = {1, 16};
Physical Volume("volume", 48) = {1};
