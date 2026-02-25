.. _conventions:

Notational Conventions
==========================
The following notational conventions are adopted throughout the code:

* All scalars are represented by lowercase unbolded letters (:math:`x`)
* Overdots on scalars represent differentiation with respect to time:

   .. math::
     \dot x \equiv \frac{\mathop{}\!\mathrm{d}{}}{\mathop{}\!\mathrm{d}{t}} \qquad \ddot x \equiv {  {\vphantom{\frac{\mathop{}\!\mathrm{d}{^{2}}}{\mathop{}\!\mathrm{d}{t^2}}}}^{\mathcal{}}\!{\frac{\mathop{}\!\mathrm{d}{^{2}}}{\mathop{}\!\mathrm{d}{t^2}}}  }x

* All matrices are represented by uppercase unbolded letters (:math:`A`)
* All vectors are represented by lowercase bolded letters (:math:`\mathbf v`). All vectors are assumed to be Euclidean (:math:`\mathbb{R}^3`)
* Unit vectors are represented by lowercase bolded letters with hats (:math:`{\hat{\mathbf{v}}}` is the unit vector of :math:`\mathbf v`)
* Vector norms are denoted by :math:`\Vert \mathbf v \Vert` such that :math:`\mathbf v = \Vert \mathbf v \Vert {\hat{\mathbf{v}}}`
* Position vectors are denoted as :math:`\mathbf r_{B/A}` indicating that this is the vector pointing from point :math:`A` to point :math:`B`
* Reference frames are represented by calligraphic capital letters (:math:`\mathcal I`) and defined by a coordinate origin and a set of three mutually orthogonal unit vectors such that  :math:`\mathcal I = ({\hat{\mathbf{e}}}_1, {\hat{\mathbf{e}}}_2, {\hat{\mathbf{e}}}_3)` implies that :math:`{\hat{\mathbf{e}}}_1\times {\hat{\mathbf{e}}}_2 = {\hat{\mathbf{e}}_3}`. All reference frames are dextral
* The component representation of any vector :math:`\mathbf v` in reference frame :math:`\mathcal I` is given by :math:`[\mathbf v]_\mathcal I` and is assumed to be a (3x1) column matrix
* Direction cosine matrices (DCMs) are defined as :math:`{{\vphantom{C}}^{\mathcal{B}}\!{C}^{\mathcal{I}}}` such that:

  .. math::
    [\mathbf v]_\mathcal B = {{\vphantom{C}}^{\mathcal{B}}\!{C}^{\mathcal{I}}}[\mathbf v]_\mathcal I
* The inverse of a direction cosine matrix is its transpose and is denoted by switching the order of the superscripts: 
  
  .. math::
    \left({{\vphantom{C}}^{\mathcal{B}}\!{C}^{\mathcal{I}}}\right)^{-1} \equiv \left({{\vphantom{C}}^{\mathcal{B}}\!{C}^{\mathcal{I}}}\right)^T = {{\vphantom{C}}^{\mathcal{I}}\!{C}^{\mathcal{B}}}
* We define the three DCMs associated with counter-clockwise (CCW) rotations about reference frame axes as:

  .. math::
    C_1(\theta)  \triangleq \begin{bmatrix} 1& 0 & 0\\ 0& \cos\theta & \sin\theta  \\ 0& -\sin\theta & \cos\theta  \end{bmatrix} \quad C_2(\theta) \triangleq \begin{bmatrix}  \cos\theta & 0 & -\sin\theta \\ 0 & 1 & 0 \\ \sin\theta & 0 & \cos\theta \end{bmatrix}\quad C_3(\theta) \triangleq \begin{bmatrix}  \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta  & 0\\0 & 0 & 1  \end{bmatrix}
* The projection of a vector :math:`\mf a` onto a plane orthogonal to direction :math:`\bhat b` is given by:

  .. math::
    \mf a_\mathrm{proj} = \mf a - (\mf a \cdot \bhat b)\bhat b

 where :math:`\cdot` represents the scalar dot product. 
* The angle between unit vectors :math:`\bhat x` and :math:`\bhat y` assuming counter-clockwise rotation about axis :math:`\bhat z` is:

  .. math::
    \theta = \operatorname{atan2}\left(\Vert \bhat x \times \bhat y \Vert\operatorname{sign}\left(\det\begin{bmatrix} \bhat x & \bhat{y} & \bhat z \end{bmatrix}\right), \bhat{x} \cdot \bhat{y}\right)

 where atan2 is the two-term arctangent operator, :math:`\det` is a matrix determinant, and sign is the mathematical sign operator. 


