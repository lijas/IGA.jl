# Bezier extraction

The basefunctions ins IGA (B-splines, NURBS, ...) exist through multiple adjecent elements. This is in contrast to traditional finite element method, where identical shape function are defined withing all elements. This makes it a bit more difficult to incorprate IGA in to a standard finite element code. The bezier extraction technique, introduced by [Borden et.al](https://doi.org/10.1002/nme.2968), solves this problem by allowing numerical integration of smooth function to be performed on C0 Bezier elements.

todo: theroy...