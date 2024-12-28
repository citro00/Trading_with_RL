��#   * * A m b i e n t e   d i   T r a d i n g   c o n   D e e p   R e i n f o r c e m e n t   L e a r n i n g * * 
 
 
 
 # #   * * P a n o r a m i c a * * 
 
 Q u e s t o   p r o g e t t o   i m p l e m e n t a   u n   a m b i e n t e   d i   t r a d i n g   p e r   s i m u l a r e   e   v a l u t a r e   s t r a t e g i e   f i n a n z i a r i e   u t i l i z z a n d o   i l   D e e p   Q - L e a r n i n g   ( D Q N )   e   i l   Q - L e a r n i n g .   I   c o m p o n e n t i   p r i n c i p a l i   i n c l u d o n o : 
 
 
 
 1 .   * * A m b i e n t e   d i   T r a d i n g   P e r s o n a l i z z a t o * * :   B a s a t o   s u   ` g y m _ a n y t r a d i n g ` ,   e s t e s o   p e r   i l   t r a d i n g   m u l t i - a s s e t . 
 
 2 .   * * A g e n t e   D Q N * * :   A g e n t e   b a s a t o   s u   r e t e   n e u r a l e   p r o f o n d a   p e r   l a   p r e s a   d i   d e c i s i o n i . 
 
 3 .   * * P r e p r o c e s s i n g   d e i   D a t i * * :   F u n z i o n i   p e r   s c a r i c a r e ,   p u l i r e   e   a r r i c c h i r e   i   d a t i   f i n a n z i a r i   d a   Y a h o o   F i n a n c e . 
 
 4 .   * * P i p e l i n e   d i   A d d e s t r a m e n t o   e   V a l u t a z i o n e * * :   R o u t i n e   a u t o m a t i z z a t e   p e r   a d d e s t r a r e   e   t e s t a r e   g l i   a g e n t i   s u   d a t i   d i   m e r c a t o   s t o r i c i . 
 
 
 
 - - - 
 
 
 
 # #   * * S t r u t t u r a   d e l   P r o g e t t o * * 
 
 
 
 # # #   * * C o m p o n e n t i   P r i n c i p a l i * * 
 
 
 
 -   * * C u s t o m S t o c k s E n v * * :   U n   a m b i e n t e   p e r s o n a l i z z a t o   p e r   i l   t r a d i n g   d i   p i �   a s s e t . 
 
 -   * * D Q N A g e n t * * :   A g e n t e   b a s a t o   s u   D e e p   Q - L e a r n i n g   p e r   a p p r e n d e r e   p o l i t i c h e   o t t i m a l i . 
 
 -   * * F u n z i o n i   U t i l i t y * * :   S t r u m e n t i   p e r   i l   d o w n l o a d ,   l a   p u l i z i a   e   l ' e l a b o r a z i o n e   d e i   d a t i . 
 
 -   * * S c r i p t   P r i n c i p a l e * * :   I n t e g r a   t u t t i   i   c o m p o n e n t i   p e r   l ' a d d e s t r a m e n t o   e   l a   v a l u t a z i o n e . 
 
 
 
 - - - 
 
 
 
 # #   * * F l u s s o   d i   L a v o r o * * 
 
 
 
 1 .   * * P r e p a r a z i o n e   d e i   D a t i * * : 
 
       -   S c a r i c a   i   d a t i   s t o r i c i   d e i   p r e z z i   u t i l i z z a n d o   Y a h o o   F i n a n c e . 
 
       -   P u l i s c i   e   p r e - e l a b o r a   i   d a t i   p e r   c a l c o l a r e   m e t r i c h e   c o m e   r e n d i m e n t o   g i o r n a l i e r o ,   r e n d i m e n t o   c u m u l a t i v o ,   S M A   e   V W A P . 
 
 2 .   * * C o n f i g u r a z i o n e   d e l l ' A m b i e n t e * * : 
 
       -   C o n f i g u r a   ` C u s t o m S t o c k s E n v `   c o n   i   d a t i   p r e - e l a b o r a t i   e   i   p a r a m e t r i   d i   t r a d i n g . 
 
 3 .   * * I n i z i a l i z z a z i o n e   d e l l ' A g e n t e * * : 
 
       -   I n i z i a l i z z a   l ' a g e n t e   D Q N   o   Q - L e a r n i n g   i n   b a s e   a l l ' i n p u t   d e l l ' u t e n t e . 
 
 4 .   * * A d d e s t r a m e n t o * * : 
 
       -   A d d e s t r a   l ' a g e n t e   p e r   p i �   e p i s o d i ,   m o n i t o r a n d o   m e t r i c h e   c o m e   p r o f i t t o   e   r i c o m p e n s a . 
 
 5 .   * * V a l u t a z i o n e * * : 
 
       -   T e s t a   l ' a g e n t e   s u   d a t i   n o n   v i s t i   e   v i s u a l i z z a   l a   p e r f o r m a n c e   d i   t r a d i n g . 
 
 
 
 - - - 
 
 
 
 # #   * * F o r m u l a z i o n i   M a t e m a t i c h e * * 
 
 
 
 # # #   * * C a l c o l o   d e l l a   R i c o m p e n s a * * 
 
 
 
 L a   f u n z i o n e   d i   r i c o m p e n s a   v a l u t a   l a   r e d d i t i v i t �   d i   o g n i   a z i o n e : 
 
 
 
 -   * * R i c o m p e n s a   d i   V e n d i t a * * : 
 
     $   R _ { s e l l }   =   \ l o g \ l e f t ( \ f r a c { P _ t } { P _ { t _ { l a s t } } } \ r i g h t )   +   c   \ t e x t {   s e   }   P _ t   >   P _ { t _ { l a s t } }   \ t e x t {   a l t r i m e n t i   }   \ l o g \ l e f t ( \ f r a c { P _ t } { P _ { t _ { l a s t } } } \ r i g h t )   -   c   $ 
 
 
 
 -   * * R i c o m p e n s a   d i   A c q u i s t o * * : 
 
     $   R _ { b u y }   =   \ l o g \ l e f t ( \ f r a c { P _ { t _ { l a s t } } } { P _ t } \ r i g h t )   +   c   \ t e x t {   s e   }   P _ t   <   P _ { t _ { l a s t } }   \ t e x t {   a l t r i m e n t i   }   \ l o g \ l e f t ( \ f r a c { P _ { t _ { l a s t } } } { P _ t } \ r i g h t )   -   c   $ 
 
 
 
 -   * * R i c o m p e n s a   d i   M a n t e n i m e n t o * * : 
 
     $   R _ { h o l d }   =   \ l o g \ l e f t ( \ f r a c { P _ t } { P _ { t _ { l a s t } } } \ r i g h t )   +   c   \ t e x t {   s e   }   P _ t   >   P _ { t _ { l a s t } }   \ t e x t {   a l t r i m e n t i   }   \ l o g \ l e f t ( \ f r a c { P _ t } { P _ { t _ { l a s t } } } \ r i g h t )   -   c   $ 
 
 
 
 D o v e : 
 
 -   $ P _ t $ :   P r e z z o   a l   t i c k   c o r r e n t e . 
 
 -   $ P _ { t _ { l a s t } } $ :   P r e z z o   a l l ' u l t i m o   t r a d e . 
 
 -   $ c $ :   C o s t a n t e   d i   b i a s   p e r   i n c e n t i v a r e   a z i o n i   p o s i t i v e . 
 
 
 
 # # #   * * R e g o l a   d i   A g g i o r n a m e n t o   d e l   Q - L e a r n i n g * * 
 
 
 
 P e r   u n o   s t a t o   \ ( s \ ) ,   u n ' a z i o n e   \ ( a \ )   e   u n a   r i c o m p e n s a   \ ( r \ ) : 
 
 $   Q ( s ,   a )   \ l e f t a r r o w   Q ( s ,   a )   +   \ a l p h a   \ l e f t [   r   +   \ g a m m a   \ m a x _ a   Q ( s ' ,   a ' )   -   Q ( s ,   a )   \ r i g h t ]   $ 
 
 
 
 D o v e : 
 
 -   $ \ \ a l p h a $ :   T a s s o   d i   a p p r e n d i m e n t o . 
 
 -   $ \ \ g a m m a $ :   F a t t o r e   d i   s c o n t o . 
 
 -   $ s ' $ :   S t a t o   s u c c e s s i v o . 
 
 -   $ a ' $ :   A z i o n e   o t t i m a l e   n e l l o   s t a t o   s u c c e s s i v o . 
 
 
 
 # # #   * * F u n z i o n e   d i   P e r d i t a   d e l   D Q N * * 
 
 
 
 L a   p e r d i t a   �   c a l c o l a t a   c o m e : 
 
 $   L ( \ t h e t a )   =   \ m a t h b b { E } \ l e f t [   \ l e f t (   r   +   \ g a m m a   \ m a x _ { a ' }   Q ( s ' ,   a ' ;   \ t h e t a ^ - )   -   Q ( s ,   a ;   \ t h e t a )   \ r i g h t ) ^ 2   \ r i g h t ]   $ 
 
 
 
 D o v e   $ \ \ t h e t a $   s o n o   i   p a r a m e t r i   d e l   m o d e l l o   o n l i n e   e   $ \ \ t h e t a ^ - $   s o n o   i   p a r a m e t r i   d e l   m o d e l l o   t a r g e t . 
 
 
 
 - - - 
 
 
 
 # #   * * D e t t a g l i   d e l l ' I m p l e m e n t a z i o n e * * 
 
 
 
 # # #   * * A m b i e n t e * * 
 
 I l   ` C u s t o m S t o c k s E n v `   �   p r o g e t t a t o   p e r : 
 
 -   G e s t i r e   p i �   a s s e t . 
 
 -   F o r n i r e   u n o   s p a z i o   d i   o s s e r v a z i o n e   c o n   d a t i   n o r m a l i z z a t i   d i   p r e z z o   e   v o l u m e . 
 
 -   C a l c o l a r e   l e   r i c o m p e n s e   b a s a t e   s u l l e   a z i o n i   d i   t r a d i n g . 
 
 
 
 # # #   * * A r c h i t e t t u r a   d e l   D Q N * * 
 
 
 
 I l   D Q N   �   c o m p o s t o   d a : 
 
 -   I n p u t :   C a r a t t e r i s t i c h e   d e l l o   s t a t o   o s s e r v a t o . 
 
 -   D u e   l a y e r   n a s c o s t i   c o n   a t t i v a z i o n i   R e L U . 
 
 -   O u t p u t :   Q - v a l o r i   p e r   o g n i   a z i o n e . 
 
 
 
 $ $   \ t e x t { Q - v a l o r i :   }   Q ( s ,   a )   \ a p p r o x   \ t e x t { R e t e   N e u r a l e } ( s ;   \ t h e t a )   $ $ 
 
 
 
 - - - 
 
 
 
 # #   * * I s t r u z i o n i   p e r   l ' U s o * * 
 
 
 
 1 .   * * I n s t a l l a   l e   D i p e n d e n z e * * : 
 
       ` ` ` b a s h 
 
       p i p   i n s t a l l   - r   r e q u i r e m e n t s . t x t 
 
       ` ` ` 
 
 
 
 2 .   * * E s e g u i   l o   S c r i p t   P r i n c i p a l e * * : 
 
       ` ` ` b a s h 
 
       p y t h o n   m a i n . p y 
 
       ` ` ` 
 
 
 
 3 .   * * S e l e z i o n a   i l   T i p o   d i   A g e n t e * * : 
 
       S c e g l i   t r a   ` D Q N `   o   ` Q L `   d u r a n t e   l ' e s e c u z i o n e . 
 
 
 
 - - - 
 
 
 
 # #   * * O u t p u t   d i   E s e m p i o * * 
 
 
 
 -   * * M e t r i c h e   d i   A d d e s t r a m e n t o * * : 
 
     -   P r o f i t t o   t o t a l e   e   r i c o m p e n s a . 
 
     -   A n d a m e n t o   d e l l a   p e r d i t a   d u r a n t e   l ' a d d e s t r a m e n t o . 
 
 
 
 -   * * V a l u t a z i o n e * * : 
 
     -   P e r f o r m a n c e   d i   t r a d i n g   v i s u a l i z z a t a   c o m e   s e g n a l i   d i   a c q u i s t o / v e n d i t a   s u l   g r a f i c o   d e i   p r e z z i . 
 
     -   P r o f i t t o   f i n a l e   e   R O I . 
 
 
 
 - - - 
 
 
 
 # #   * * R i f e r i m e n t i * * 
 
 1 .   [ G y m - A n y t r a d i n g ] ( h t t p s : / / g i t h u b . c o m / A m i n H P / g y m - a n y t r a d i n g ) 
 
 2 .   [ D e e p   Q - L e a r n i n g   P a p e r ] ( h t t p s : / / a r x i v . o r g / a b s / 1 3 1 2 . 5 6 0 2 ) 
 
 3 .   [ Y a h o o   F i n a n c e   A P I ] ( h t t p s : / / p y p i . o r g / p r o j e c t / y f i n a n c e / ) 
 
 
 
 - - - 
 
 
 
 # #   * * C o n t a t t i * * 
 
 P e r   d o m a n d e   o   c o n t r i b u t i ,   a p r i   u n   i s s u e   o   i n v i a   u n a   p u l l   r e q u e s t . 
 
 