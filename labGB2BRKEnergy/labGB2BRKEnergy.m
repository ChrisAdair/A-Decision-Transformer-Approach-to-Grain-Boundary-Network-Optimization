%-------------------------------------------------------------------------%
% Filename: labGB2BRKEnergy.m
% Author: Oliver Johnson
% Date: 8/27/2024
%
% labGB2BRKEnergy will calculate GB energy for a user specified material
% using the BRK function (GB5DOF.m) given the orientations of the incident
% grains (expressed as unit quaternions) and the GB normal (expressed as a 
% unit vector), all expressed in the same laboratory frame.
%
% NOTE: Olmsted's GB matrices are defined such that the GB plane lies in
% the z = 0 plane of the reference frame. However, the GB5DOF matrices are
% defined such that the GB plane lies in the x = 0 plane of the reference
% frame. Consequently, we provide for calculation of the matrices using
% either convention.
%
% Inputs:
%   qA_Lab - A 1-by-4 array containing a single unit quaternion 
%            representing the orientation of one of the grains at the GB 
%            (grain A). With a reference coordinate system defined 
%            such that z = 0 is the GB plane, grain A is the region z < 0.
%   qB_Lab - A 1-by-4 array containing a single unit quaternion 
%            representing the orientation of one of the grains at the GB 
%            (grain B). With a reference coordinate system defined 
%            such that z = 0 is the GB plane, grain B is the region z > 0.
%   nA_Lab - A 3-by-1 array containing a single 3D unit vector representing
%            the GB normal expressed in the Lab coordinate system. As a 
%            convention, we define nA_Lab such that it is pointing away 
%            from grain A (i.e. it is the outward-pointing normal).
%   material - A string indicating which material to calculate the GB
%              energy for. As indicated in the GB5DOF.m documentation, 
%              allowed values are 'Al', 'Ni', 'Au', or 'Cu'. %                   
%
% Outputs:
%   E - A scalar representing the GB energy as calculated using the BRK
%       function (GB5DOF.m) for the user specified material. Units will be
%       in J/m^2.
%
% [1] Olmsted, D. L. (2009). A new class of metrics for the macroscopic 
%     crystallographic space of grain boundaries. Acta Materialia, 57(9), 
%     2793–2799. https://doi.org/10.1016/j.actamat.2009.02.030
% [2] Bulatov, V. V, Reed, B. W., & Kumar, M. (2014). Grain boundary energy 
%     function for fcc metals. Acta Materialia, 65, 161–175. 
%     https://doi.org/10.1016/j.actamat.2013.10.057
%-------------------------------------------------------------------------%

function E = labGB2BRKEnergy(qA_Lab,qB_Lab,nA_Lab,material)

% Compute GB matrices
[gA_R,gB_R] = constructGBMatrices(qA_Lab,qB_Lab,nA_Lab,'livermore');

% Calculate GB energy
E = GB5DOF(gA_R,gB_R,material);

end

%-------------------------------------------------------------------------%
% Filename: constructGBMatrices.m
% Author: Oliver Johnson
% Date: 3/11/2020
%
% CONSTRUCTGBMATRICES will convert the crystal orientations (expressed as 
% quaternions) of grains meeting at a grain boundary (GB) together with the
% corresponding GB normals to the pair of GB matrices defined by Olmsted
% [1] and required by the GB5DOF function [2].
%
% NOTE: Olmsted's GB matrices are defined such that the GB plane lies in
% the z = 0 plane of the reference frame. However, the GB5DOF matrices are
% defined such that the GB plane lies in the x = 0 plane of the reference
% frame. Consequently, we provide for calculation of the matrices using
% either convention.
%
% Inputs:
%   qA_Lab - An nGB-by-4 array of quaternions representing the orientations 
%            of one of the grains at the GB (grain A). qA_Lab(i,:) 
%            represents the quaternion defining the orientatino of grain A 
%            for the i-th GB. With a reference coordinate system defined 
%            such that z = 0 is the GB plane, grain A is the region z < 0.
%   qB_Lab - An nGB-by-4 array of quaternions representing the orientations 
%            of one of the grains at the GB (grain B). qB_Lab(i,:) 
%            represents the quaternion defining the orientatino of grain B 
%            for the i-th GB. With a reference coordinate system defined 
%            such that z = 0 is the GB plane, grain B is the region z > 0.
%   nA_Lab - A 3-by-nGB array of 3D vectors representing the GB normals
%            expressed in the Lab coordinate system. As a convention, we
%            define nA_Lab such that it is pointing away from grain A (i.e.
%            it is the outward-pointing normal). nA_Lab(:,i) give the
%            vector representing the GB normal for the i-th GB.
%
% Outputs:
%   gA_R - A 3-by-3-by-nGB array of rotation matrices representing the
%          orientation of grain A expressed in the reference coordinate
%          system which has the z = 0 plane aligned with the GB plane.
%          gA_R(:,:,i) represents the orientation of grain A for the i-th
%          GB.
%   gB_R - A 3-by-3-by-nGB array of rotation matrices representing the
%          orientation of grain B expressed in the reference coordinate
%          system which has the z = 0 plane aligned with the GB plane.
%          gB_R(:,:,i) represents the orientation of grain B for the i-th
%          GB.
%
%   NOTE: When using the 'livermore' convention, gA_R is defined such that 
%   gA_R(1,:,i) = nA_A(:,i).' (i.e. the 1st row of gA_R corresponds to the 
%   GB normal pointing away from grain A, expressed in the coordinate 
%   system of grain A, with nA_A(:,i) = gA_Lab(:,:,i).'*nA_Lab(:,i)).
%   Meanwhile, gA_B is defined such that gB_R(1,:,i) = -nB_B(:,i).' (i.e. 
%   the 1st row of gB_R corresponds to the GB normal pointing away from 
%   grain B, expressed in the coordinate system of grain B, with 
%   nB_B(:,i) = gB_Lab(:,:,i).'*nB_Lab(:,i)).
%
%   NOTE: The inputs P and Q for GB5DOF correspond, respectively to gA_R
%   and gB_R. It appears that the convention used in the GB5DOF code is to
%   define a single GB normal, which points away from grain B. This is why
%   the documentation for GB5DOF says that Nq = Q(1,:), whereas by our
%   definition gB_R(1,:,i) = -nB_B(:,i).' (Nq = -Q(1,:)).
%
% [1] Olmsted, D. L. (2009). A new class of metrics for the macroscopic 
%     crystallographic space of grain boundaries. Acta Materialia, 57(9), 
%     2793–2799. https://doi.org/10.1016/j.actamat.2009.02.030
% [2] Bulatov, V. V, Reed, B. W., & Kumar, M. (2014). Grain boundary energy 
%     function for fcc metals. Acta Materialia, 65, 161–175. 
%     https://doi.org/10.1016/j.actamat.2013.10.057
%-------------------------------------------------------------------------%

function [gA_R,gB_R] = constructGBMatrices(qA_Lab,qB_Lab,nA_Lab,convention)

%% Ensure proper formatting of inputs

assert(size(qA_Lab,2) == 4 && size(qB_Lab,2) == 4,'qA_Lab and qB_Lab must be n-by-4 arrays of quaternions.')
assert(size(nA_Lab,1) == 3,'nA_Lab must be a 3-by-n array of vectors.')

% number of GBs
Ngb = size(qA_Lab,1);

% ensure GB normals are normalized
nA_Lab = nA_Lab./sqrt(sum(nA_Lab.^2,1));

% normals pointing away from grain B in the Lab frame
nB_Lab = -nA_Lab;

%% Construct reference frame quaternions (qR)

switch convention
    case 'olmsted' % olmsted convention has z-axis aligned to -nB_Lab
        zR = -nB_Lab;
        
        xR = cross(zR,rand(3,Ngb)); % cross with random vector to get an arbitrary vector in the GB plane to use as the x-axis
        xR = xR./sqrt(sum(xR.^2,1));
    case 'livermore' % livermore convention has x-axis aligned to -nB_Lab
        xR = -nB_Lab;
        
        zR = cross(xR,rand(3,Ngb)); % cross with random vector to get an arbitrary vector in the GB plane to use as the x-axis
        zR = zR./sqrt(sum(zR.^2,1));
end

yR = cross(zR,xR); % cross z with x to define y-axis
yR = yR./sqrt(sum(yR.^2,1));

% make rotation matrices
gR = [permute(xR,[1,3,2]),permute(yR,[1,3,2]),permute(zR,[1,3,2])];

% convert to quaternions
qR = gmat2q(gR);

%% Construct qA_R and qB_R (quaternions in the reference frame)

qA_R = qmultiply(qinv(qR),qA_Lab);
qB_R = qmultiply(qinv(qR),qB_Lab);

%% Convert to rotation matrices

gA_R = q2gmat(qA_R);
gB_R = q2gmat(qB_R);

% NOTE: P = gA_R; Q = gB_R;

end

%% INSERT GB5DOF CODE HERE
% Source: http://dx.doi.org/10.1016/j.actamat.2013.10.057
% function en = GB5DOF(P,Q,AlCuParameter,eRGB)
% 
% end

%%
function q = gmat2q(g)
%-------------------------------------------------------------------------%
%Filename:  gmat2q.m
%Author:    Oliver Johnson
%Date:      3/28/2020
%
% Inputs:
%   g - A 3-by-3-by-npts array of rotation matrices.
%
% Outputs:
%   q - An npts-by-4 array of equiavalent quaternions.
%
% CURRENT METHOD
% Shepperd's Method (as described in
% https://doi.org/10.1007/978-3-319-93188-3_5). Works correctly for both
% skew-symmetric and symmetric (e.g. rotations by pi) rotation matrices.
% 
% OLD METHOD
% https://doi.org/10.1007/978-3-319-93188-3_5 (Only works for
% skew-symmetric rotation matrices)
%
% OLD OLD METHOD (4/1/2013):
% Code adapted from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
% This algorithm only works when the rotation matrix is stricly skew
% symmetric, in the case that it is symmetric the difference of the off
% diagonal terms is zero so the sign function returns zero. I need to get a
% better method to assign the signs consistently. Take a look at MTEX's
% mat2quat function.
%-------------------------------------------------------------------------%

%---pre-allocate---%
npts = size(g,3);
q = zeros(npts,4);

%---compute transformation---%
% compute the traces
t = g(1,1,:)+g(2,2,:)+g(3,3,:);

% figure out which term is largest (this will determine which case to use)
[~,id] = max([t,g(1,1,:),g(2,2,:),g(3,3,:)],[],2);

% case 1
id1 = id == 1;
val = sqrt(1+t(:,:,id1));
q(id1,1) = 0.5*squeeze(val);
q(id1,2) = 0.5*squeeze(( g(3,2,id1) - g(2,3,id1) ) ./ val);
q(id1,3) = 0.5*squeeze(( g(1,3,id1) - g(3,1,id1) ) ./ val);
q(id1,4) = 0.5*squeeze(( g(2,1,id1) - g(1,2,id1) ) ./ val);

% case 2
id2 = id == 2;
val = sqrt(1+g(1,1,id2)-g(2,2,id2)-g(3,3,id2));
q(id2,1) = 0.5*squeeze(( g(3,2,id2) - g(2,3,id2) ) ./ val);
q(id2,2) = 0.5*squeeze(val);
q(id2,3) = 0.5*squeeze(( g(1,2,id2) + g(2,1,id2) ) ./ val);
q(id2,4) = 0.5*squeeze(( g(3,1,id2) + g(1,3,id2) ) ./ val);

% case 3
id3 = id == 3;
val = sqrt(1-g(1,1,id3)+g(2,2,id3)-g(3,3,id3));
q(id3,1) = 0.5*squeeze(( g(1,3,id3) - g(3,1,id3) ) ./ val);
q(id3,2) = 0.5*squeeze(( g(1,2,id3) + g(2,1,id3) ) ./ val);
q(id3,3) = 0.5*squeeze(val);
q(id3,4) = 0.5*squeeze(( g(2,3,id3) + g(3,2,id3) ) ./ val);

% case 4
id4 = id == 4;
val = sqrt(1-g(1,1,id4)-g(2,2,id4)+g(3,3,id4));
q(id4,1) = 0.5*squeeze(( g(2,1,id4) - g(1,2,id4) ) ./ val);
q(id4,2) = 0.5*squeeze(( g(3,1,id4) + g(1,3,id4) ) ./ val);
q(id4,3) = 0.5*squeeze(( g(3,2,id4) + g(2,3,id4) ) ./ val);
q(id4,4) = 0.5*squeeze(val);

% choose overall sign that makes q0 positive
idChangeSign = q(:,1) < 0;
q(idChangeSign,:) = -q(idChangeSign,:);

end

%-------------------------------------------------------------------------%
%Filename:  qinv.m
%Author:    Oliver Johnson
%Date:      2/23/2011
%
% QINV takes the inverse of quaternions.
%
% Inputs:
%   q - An npts-by-4 array containing quaternion components in 4-vector 
%       format.
%
% Outputs:
%   qi - An npts-by-1 vector containing the inverse of each of the npts
%        quaternions in q, i.e., qi(i) = q(i,:)#-1, where #-1 denotes the
%        operation of quaternion inversion.
%-------------------------------------------------------------------------%

function qi = qinv(q)

%---check inputs---%
assert((size(q,2) == 4),'q must be an npts-by-4 array.')

%---perform quaternion inversion---%
a = qnorm(q).^-2;
a = a(:,ones(4,1));
qi = a.*qconj(q);

%---correct for numerical errors---%
qi(abs(qi) < eps) = 0;

end

%-------------------------------------------------------------------------%
%Filename:  qnorm.m
%Author:    Oliver Johnson
%Date:      2/23/2011
%
% QNORM returns the magnitude (2-norm) of user supplied quaternions.
%
% Inputs:
%   q - An npts-by-4 array containing quaternion components in 4-vector 
%        format.
%
% Outputs:
%   qn - An npts-by-1 vector containing the norm of each of the npts
%        quaternions in q, i.e., qn(i) = norm(q(i,:)).
%-------------------------------------------------------------------------%

function qn = qnorm(q)

%---check inputs---%
assert((size(q,2) == 4),'q must be an npts-by-4 array.')

%---compute quaternion magnitudes---%
qn = sqrt(sum(q.^2,2));

end

%-------------------------------------------------------------------------%
%Filename:  qmultiply.m
%Author:    Oliver Johnson
%Date:      2/23/2011
%
% QMULTIPLY performs quaternion multiplication.
%
% Inputs:
%   qa - An npts-by-4 array containing quaternion components in 4-vector 
%        format.
%   qb - An npts-by-4 array containing quaternion components in 4-vector 
%        format.
%
% Outputs:
%   qc - An npts-by-4 array containing the quaternion components resulting
%        from multiplication of quaternions in qa with quaternions in qb by
%        row, i.e., qc(i,:) = qa(i,:)#qb(i,:), where # is here used to
%        denote quaternion multiplication.
%-------------------------------------------------------------------------%

function qc = qmultiply(qa,qb)

%---check inputs---%
assert((size(qa,2) == 4),'qa must be an npts-by-4 array.')
assert((size(qb,2) == 4),'qb must be an npts-by-4 array.')
assert((size(qa,1) == size(qb,1)),'qa and qb must have the same number of points.')

%---perform quaternion multiplication---%
qc(:,1) = qb(:,1).*qa(:,1)-qb(:,2).*qa(:,2)-qb(:,3).*qa(:,3)-qb(:,4).*qa(:,4);
qc(:,2) = qb(:,2).*qa(:,1)+qb(:,1).*qa(:,2)+qb(:,4).*qa(:,3)-qb(:,3).*qa(:,4);
qc(:,3) = qb(:,3).*qa(:,1)-qb(:,4).*qa(:,2)+qb(:,1).*qa(:,3)+qb(:,2).*qa(:,4);
qc(:,4) = qb(:,4).*qa(:,1)+qb(:,3).*qa(:,2)-qb(:,2).*qa(:,3)+qb(:,1).*qa(:,4);

end

%-------------------------------------------------------------------------%
%Filename:  qconj.m
%Author:    Oliver Johnson
%Date:      2/23/2011
%
% QCONJ takes the conjugate of quaternions.
%
% Inputs:
%   q - An npts-by-4 array containing quaternion components in 4-vector 
%       format.
%
% Outputs:
%   q_star - An npts-by-4 array containing the conjugate of each of the 
%            npts quaternions in q, i.e., q_star(i,:) = qconj(q(i,:)).
%-------------------------------------------------------------------------%

function q_star = qconj(q)

%---check inputs---%
assert((size(q,2) == 4),'q must be an npts-by-4 array.')

%---compute conjugates of quaternions---%
q_star = [q(:,1), -q(:,2), -q(:,3), -q(:,4)];

end

%-------------------------------------------------------------------------%
%Filename:  q2gmat.m
%Author:    Oliver Johnson
%Date:      6/2/2011
%
% Q2GMAT converts quaternions to their canonical 3x3 rotation matrix
% representation, using the Euler-Rodrigues formula.
%
% Inputs:
%   q - An npts-by-4 matrix of quaternion components.
%
% Outputs:
%   g - A 3-by-3-by-npts array of rotation matrices, where g(:,:,i) gives
%       the rotation matrix for q(i,:).
%-------------------------------------------------------------------------%

function g = q2gmat(q)

assert(size(q,2) == 4,'q must be an npts-by-4 array.')

%---determine the number of quaternions given---%
npts = size(q,1);

%---extract quaternion components---%
q0 = q(:,1);
q1 = q(:,2);
q2 = q(:,3);
q3 = q(:,4);

%---initialize g-matrix array---%
g = zeros(3,3,npts);

%---compute elements of g-matrices---% %consider using instead, the formula from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
g(1,1,:) = reshape(q0.^2+q1.^2-q2.^2-q3.^2,[1 1 npts]);
g(1,2,:) = reshape(2.*(q1.*q2-q0.*q3),[1 1 npts]);
g(1,3,:) = reshape(2.*(q1.*q3+q0.*q2),[1 1 npts]);
g(2,1,:) = reshape(2.*(q1.*q2+q0.*q3),[1 1 npts]);
g(2,2,:) = reshape(q0.^2-q1.^2+q2.^2-q3.^2,[1 1 npts]);
g(2,3,:) = reshape(2.*(q2.*q3-q0.*q1),[1 1 npts]);
g(3,1,:) = reshape(2.*(q1.*q3-q0.*q2),[1 1 npts]);
g(3,2,:) = reshape(2.*(q2.*q3+q0.*q1),[1 1 npts]);
g(3,3,:) = reshape(q0.^2-q1.^2-q2.^2+q3.^2,[1 1 npts]);

end