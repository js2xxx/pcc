macro_rules! __define_point {
    (@ORIG, $type:ident < $data:ident, $num:literal >) => {
        #[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Default)]
        #[repr(align(16))]
        pub struct $type(SVector<$data, $num>);

        impl Point for $type {
            type Data = $data;
            const DIM: usize = $num;

            fn coords(&self) -> MatrixSlice<$data, Const<4>, Const<1>, Const<1>, Const<$num>> {
                self.0.fixed_rows(0)
            }

            fn coords_mut(
                &mut self,
            ) -> MatrixSliceMut<$data, Const<4>, Const<1>, Const<1>, Const<$num>> {
                self.0.fixed_rows_mut(0)
            }

            fn as_slice(&self) -> &[$data] {
                self.0.as_slice()
            }

            fn as_mut_slice(&mut self) -> &mut [$data] {
                self.0.as_mut_slice()
            }

            fn with_coords(coords: &Vector4<$data>) -> Self {
                Self(coords.insert_fixed_rows::<{ $num - 4 }>(4, 0.))
            }
        }
    };
    (rgba $get:ident: $trait:ident, $type:ident < $data:ident, $num:literal > , $index:literal) => {
        impl $trait for $type {
            fn rgb_value(&self) -> $data {
                self.0[$index]
            }

            fn set_rgb_value(&mut self, value: $data) {
                self.0[$index] = value;
            }

            fn $get(&self) -> u32 {
                self.0[$index].to_bits()
            }

            fn set_rgba(&mut self, rgba: u32) {
                self.0[$index] = $data::from_bits(rgba);
            }

            fn fields() -> array::IntoIter<FieldInfo, 1> {
                [FieldInfo::single("rgb rgba", $index)].into_iter()
            }
        }
    };
    (
        normal $get:ident: $trait:ident,
        $type:ident <
        $data:ident,
        $num:literal > ,
        $normal_index:literal,
        $curvature_index:literal
    ) => {
        impl $trait for $type {
            fn $get(&self) -> MatrixSlice<$data, Const<4>, Const<1>, Const<1>, Const<$num>> {
                self.0.fixed_rows($normal_index)
            }

            fn normal_mut(
                &mut self,
            ) -> MatrixSliceMut<$data, Const<4>, Const<1>, Const<1>, Const<$num>> {
                self.0.fixed_rows_mut($normal_index)
            }

            fn curvature(&self) -> $data {
                self.0[$curvature_index]
            }

            fn set_curvature(&mut self, curvature: $data) {
                self.0[$curvature_index] = curvature;
            }

            fn fields() -> array::IntoIter<FieldInfo, 2> {
                [
                    FieldInfo::dim3("normal", $normal_index),
                    FieldInfo::single("curvature", $curvature_index),
                ]
                .into_iter()
            }
        }
    };
    (intensity $get:ident: $trait:ident, $type:ident < $data:ident, $num:literal > , $index:literal) => {
        impl $trait for $type {
            fn $get(&self) -> $data {
                self.0[$index]
            }

            fn set_intensity(&mut self, intensity: $data) {
                self.0[$index] = intensity;
            }

            fn fields() -> array::IntoIter<FieldInfo, 1> {
                [FieldInfo::single("intensity", $index)].into_iter()
            }
        }
    };
    (label $get:ident: $trait:ident, $type:ident < $data:ident, $num:literal > , $index:literal) => {
        impl $trait for $type {
            fn $get(&self) -> u32 {
                self.0[$index].to_bits()
            }

            fn set_label(&mut self, label: u32) {
                self.0[$index] = $data::from_bits(label)
            }

            fn fields() -> array::IntoIter<FieldInfo, 1> {
                [FieldInfo::single("label", $index)].into_iter()
            }
        }
    };
    (range $get:ident: $trait:ident, $type:ident < $data:ident, $num:literal > , $index:literal) => {
        impl $trait for $type {
            fn $get(&self) -> $data {
                self.0[$index]
            }

            fn set_range(&mut self, range: $data) {
                self.0[$index] = range;
            }

            fn fields() -> array::IntoIter<FieldInfo, 1> {
                [FieldInfo::single("range", $index)].into_iter()
            }
        }
    };
    (viewpoint $get:ident: $trait:ident, $type:ident < $data:ident, $num:literal > , $index:literal) => {
        impl $trait for $type {
            fn $get(&self) -> MatrixSlice<$data, Const<4>, Const<1>, Const<1>, Const<$num>> {
                self.0.fixed_rows($index)
            }

            fn viewpoint_mut(
                &mut self,
            ) -> MatrixSliceMut<$data, Const<4>, Const<1>, Const<1>, Const<$num>> {
                self.0.fixed_rows_mut($index)
            }

            fn fields() -> array::IntoIter<FieldInfo, 1> {
                [FieldInfo::dim3("viewpoint", $index)].into_iter()
            }
        }
    };
    {
        $type:ident<$data:ident, $num:literal>
        $({ $($field:ident: $trait:ident[$($index:literal),* $(,)?]),* $(,)? })?
    } => {
        __define_point!(@ORIG, $type<$data, $num>);
        $($(
            $(const_assert!($index < $num);)*
            __define_point!($field $field: $trait, $type<$data, $num>, $($index),*);
        )*)?

        impl PointFields for $type {
            type Iter = impl Iterator<Item = FieldInfo>;

            fn fields() -> Self::Iter {
                <$type as Point>::fields()
                    $($(.chain(<$type as $trait>::fields()))*)?
            }
        }
    };
    {
        #[auto_centroid]
        $type:ident<$data:ident, $num:literal>
        $({ $($field:ident: $trait:ident[$($index:literal),* $(,)?]),* $(,)? })?
    } => {
        __define_point!($type<$data, $num> $({ $($field: $trait [$($index),*]),* })?);

        impl Centroid for $type {
            type Accumulator = (Vector4<$data>, ($($(<$type as $trait>::CentroidAccumulator,)*)?));
            type Result = Self;

            fn accumulate(&self, accum: &mut Self::Accumulator) {
                accum.0 += self.coords();
                $($(<Self as $trait>::centroid_accumulate(self, &mut accum.1. ${index()}));*)?
            }

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
        pub struct $type:ident<$data:ident, $num:literal>
        $({ $($field:ident: $trait:ident[$($index:literal),* $(,)?]),* $(,)? })? $(;)?
    )*} => {
        $(__define_point!(
            $(#[$meta])?
            $type<$data, $num>
            $({ $($field: $trait[$($index),*]),* })?
        );)*
    };
}