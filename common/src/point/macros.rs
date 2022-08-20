macro_rules! __define_point {
    (@ORIG, $type:ident < $data:ident, $num:ident >) => {
        #[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Default)]
        #[repr(align(16))]
        pub struct $type(SVector<$data, { <$num>::USIZE }>);

        impl From<$type> for SVector<$data, { <$num>::USIZE }> {
            fn from(s: $type) -> Self {
                s.0
            }
        }

        impl Point for $type {
            type Data = $data;
            type Dim = $num;

            #[inline]
            fn coords(&self) -> &Vector4<$data> {
                unsafe { &*(self.0.data.ptr() as *const _) }
            }

            #[inline]
            fn coords_mut(&mut self) -> &mut Vector4<$data> {
                unsafe { &mut *(self.0.data.ptr_mut() as *mut _) }
            }

            #[inline]
            fn as_slice(&self) -> &[$data] {
                self.0.as_slice()
            }

            #[inline]
            fn as_mut_slice(&mut self) -> &mut [$data] {
                self.0.as_mut_slice()
            }
        }
    };
    (rgba $get:ident: $trait:ident, $type:ident < $data:ident, $num:ident > , $index:literal) => {
        impl $trait for $type {
            #[inline]
            fn rgb_value(&self) -> $data {
                self.0[$index]
            }

            #[inline]
            fn set_rgb_value(&mut self, value: $data) {
                self.0[$index] = value;
            }

            #[inline]
            fn $get(&self) -> u32 {
                self.0[$index].to_bits()
            }

            #[inline]
            fn set_rgba(&mut self, rgba: u32) {
                self.0[$index] = $data::from_bits(rgba);
            }

            #[inline]
            fn fields() -> array::IntoIter<FieldInfo, 1> {
                [FieldInfo::single("rgb rgba", $index)].into_iter()
            }
        }
    };
    (
        normal $get:ident: $trait:ident,
        $type:ident <
        $data:ident,
        $num:ident > ,
        $normal_index:literal,
        $curvature_index:literal
    ) => {
        impl $trait for $type {
            #[inline]
            fn $get(&self) -> &Vector4<$data> {
                unsafe { &*(self.0.fixed_rows::<4>($normal_index).data.ptr() as *const _) }
            }

            #[inline]
            fn normal_mut(&mut self) -> &mut Vector4<$data> {
                unsafe { &mut *(self.0.fixed_rows_mut::<4>($normal_index).data.ptr_mut() as *mut _) }
            }

            #[inline]
            fn curvature(&self) -> $data {
                self.0[$curvature_index]
            }

            #[inline]
            fn curvature_mut(&mut self) -> &mut $data {
                &mut self.0[$curvature_index]
            }

            #[inline]
            fn fields() -> array::IntoIter<FieldInfo, 2> {
                [
                    FieldInfo::dim3("normal", $normal_index),
                    FieldInfo::single("curvature", $curvature_index),
                ]
                .into_iter()
            }
        }
    };
    (intensity $get:ident: $trait:ident, $type:ident < $data:ident, $num:ident > , $index:literal) => {
        impl $trait for $type {
            #[inline]
            fn $get(&self) -> $data {
                self.0[$index]
            }

            #[inline]
            fn set_intensity(&mut self, intensity: $data) {
                self.0[$index] = intensity;
            }

            #[inline]
            fn fields() -> array::IntoIter<FieldInfo, 1> {
                [FieldInfo::single("intensity", $index)].into_iter()
            }
        }
    };
    (label $get:ident: $trait:ident, $type:ident < $data:ident, $num:ident > , $index:literal) => {
        impl $trait for $type {
            #[inline]
            fn $get(&self) -> u32 {
                self.0[$index].to_bits()
            }

            #[inline]
            fn set_label(&mut self, label: u32) {
                self.0[$index] = $data::from_bits(label)
            }

            #[inline]
            fn fields() -> array::IntoIter<FieldInfo, 1> {
                [FieldInfo::single("label", $index)].into_iter()
            }
        }
    };
    (range $get:ident: $trait:ident, $type:ident < $data:ident, $num:ident > , $index:literal) => {
        impl $trait for $type {
            #[inline]
            fn $get(&self) -> $data {
                self.0[$index]
            }

            #[inline]
            fn range_mut(&mut self) -> &mut $data {
                &mut self.0[$index]
            }

            #[inline]
            fn fields() -> array::IntoIter<FieldInfo, 1> {
                [FieldInfo::single("range", $index)].into_iter()
            }
        }
    };
    (viewpoint $get:ident: $trait:ident, $type:ident < $data:ident, $num:ident > , $index:literal) => {
        impl $trait for $type {
            #[inline]
            fn $get(&self) -> &Vector4<$data> {
                unsafe { &*(self.0.fixed_rows::<4>($index).data.ptr() as *const _) }
            }

            #[inline]
            fn viewpoint_mut(&mut self) -> &mut Vector4<$data>{
                unsafe { &mut *(self.0.fixed_rows_mut::<4>($index).data.ptr_mut() as *mut _) }
            }

            #[inline]
            fn fields() -> array::IntoIter<FieldInfo, 1> {
                [FieldInfo::dim3("viewpoint", $index)].into_iter()
            }
        }
    };
    {
        $type:ident<$data:ident, $num:ident>
        $({ $($field:ident: $trait:ident[$($index:literal),* $(,)?]),* $(,)? })?
    } => {
        __define_point!(@ORIG, $type<$data, $num>);
        $($(
            $(const_assert!($index < <$num>::USIZE);)*
            __define_point!($field $field: $trait, $type<$data, $num>, $($index),*);
        )*)?

        impl PointFields for $type {
            type Iter = impl Iterator<Item = FieldInfo> + Clone;

            #[inline]
            fn fields() -> Self::Iter {
                <$type as Point>::fields()
                    $($(.chain(<$type as $trait>::fields()))*)?
            }
        }
    };
    {
        #[auto_centroid]
        $type:ident<$data:ident, $num:ident>
        $({ $($field:ident: $trait:ident[$($index:literal),* $(,)?]),* $(,)? })?
    } => {
        __define_point!($type<$data, $num> $({ $($field: $trait [$($index),*]),* })?);

        impl Centroid for $type {
            type Accumulator = (Vector4<$data>, ($($(<$type as $trait>::CentroidAccumulator,)*)?));
            type Result = Self;

            #[inline]
            fn accumulate(&self, accum: &mut Self::Accumulator) {
                accum.0 += self.coords();
                $($(<Self as $trait>::centroid_accumulate(self, &mut accum.1. ${index()}));*)?
            }

            #[inline]
            fn compute(accum: Self::Accumulator, num: usize) -> Self::Result {
                let mut result = Self::Result::default();
                result.coords_mut().set_column(0, &(accum.0 / (num as $data)));
                $($(
                    <Self::Result as $trait>::centroid_compute(
                        &mut result,
                        { accum.1. ${index()} },
                        num
                    );
                )*)?
                result
            }
        }
    };
}

macro_rules! define_points {
    {$(
        $(#[$meta:tt])?
        pub struct $type:ident<$data:ident, $num:ident>
        $({ $($field:ident: $trait:ident[$($index:literal),* $(,)?]),* $(,)? })? $(;)?
    )*} => {
        $(__define_point!(
            $(#[$meta])?
            $type<$data, $num>
            $({ $($field: $trait[$($index),*]),* })?
        );)*
    };
}
