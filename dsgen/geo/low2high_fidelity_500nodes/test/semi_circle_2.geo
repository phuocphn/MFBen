// Gmsh project created on Fri Apr 05 21:01:20 2024
SetFactory("OpenCASCADE");
mesh_size0=0.0045 * 100;
scale=1;
R =0.8*0.5*Pi*scale; 
cx = 2*Pi*scale;
cy = 2*Pi*scale;

Point(1) = {Pi * scale, Pi * scale, 0, mesh_size0};
Point(2) = {Pi * scale, 3*Pi*scale, 0, mesh_size0};
Point(3) = {3*Pi*scale, 3*Pi*scale, 0, mesh_size0};
Point(4) = {3*Pi*scale, Pi * scale, 0, mesh_size0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

//https://storage.googleapis.com/tb-img/production/18/11/Electrician_34_18_8.PNG
Point(5) = {cx - R, cy - (4*R)/(3*Pi), 0, mesh_size0};
Point(6) = {cx, cy - (4*R)/(3*Pi), 0,  mesh_size0};
Point(7) = {cx + R, cy - (4*R)/(3*Pi), 0, mesh_size0};

Circle(5) = {5, 6, 7};
Characteristic Length {5} = mesh_size0;

Line(6) = {5, 6};
Line(7) = {6, 7};

Rotate {{0, 0, 1}, {2*Pi*scale, 2*Pi*scale, 0}, Pi} {
  Curve{5}; Curve{7}; Curve{6}; 
}
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {6, 7, -5};
Plane Surface(1) = {1, 2};

Extrude {0, 0, 0.005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}
//+
Physical Surface("inlet", 22) = {2};
Physical Surface("outlet", 23) = {4};
Physical Surface("obstacle", 24) = {8, 7, 6};
Physical Surface("wall", 25) = {5, 3};
Physical Surface("frontAndBack", 26) = {9, 1};
Physical Volume("volume", 27) = {1};
