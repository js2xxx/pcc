macro_rules! impl_pi {
    (@INNER, $trait:ident$(<$scalar:ident>)?: ($func:ident, $func_mut:ident) => $ty:ident; ; ) => {
        pub trait $trait $(<$scalar: Scalar>)? {
            fn $func(&self) -> &$ty $(<$scalar>)?;

            fn $func_mut(&mut self) -> &mut $ty $(<$scalar>)?;

            fn from(value: $ty $(<$scalar>)?) -> Self where Self: Default;
        }

        impl $(<$scalar: Scalar>)? $trait$(<$scalar>)? for $ty$(<$scalar>)? {
            fn $func(&self) -> &$ty$(<$scalar>)? {
                self
            }

            fn $func_mut(&mut self) -> &mut $ty$(<$scalar>)? {
                self
            }

            fn from(value: $ty$(<$scalar>)?) -> Self where Self: Default {
                value
            }
        }
    };

    (@INNER, $trait:ident$(<$scalar:ident>)?: ($func:ident, $func_mut:ident) => $ty:ident; $($before:ident),*; $($after:ident),*) => {
        impl <$($before,)* $($scalar: Scalar,)? $($after),*>
            $trait$(<$scalar>)? for ($($before,)* $ty$(<$scalar>)?, $($after),*) {
            fn $func(&self) -> &$ty$(<$scalar>)? {
                &self. ${count(before)}
            }

            fn $func_mut(&mut self) -> &mut $ty$(<$scalar>)? {
                &mut self. ${count(before)}
            }

            fn from(value: $ty$(<$scalar>)?) -> Self where Self: Default {
                let mut ret = Self::default();
                ret. ${count(before)} = value;
                ret
            }
        }
    };

    ($trait:ident$(<$scalar:ident>)?: ($func:ident, $func_mut:ident) => $ty:ident; ; ) => {
        impl_pi!(@INNER, $trait$(<$scalar>)?: ($func, $func_mut) => $ty; ; );
    };


    ($trait:ident$(<$scalar:ident>)?: ($func:ident, $func_mut:ident) => $ty:ident; $before0:ident$(,$($before1:ident),*)?; ) => {
        impl_pi!(@INNER, $trait$(<$scalar>)?: ($func, $func_mut) => $ty; $before0$(,$($before1),*)? ; );

        impl_pi!($trait$(<$scalar>)?: ($func, $func_mut) => $ty; $($($before1),*)? ; );
    };

    ($trait:ident$(<$scalar:ident>)?: ($func:ident, $func_mut:ident) => $ty:ident; $($before:ident),*; $after0:ident$(,$($after1:ident),*)?) => {
        impl_pi!(@INNER, $trait$(<$scalar>)?: ($func, $func_mut) => $ty; $($before),*; $after0$(,$($after1),*)?);

        impl_pi!($trait$(<$scalar>)?: ($func, $func_mut) => $ty; $($before),*; $($($after1),*)?);
    };
}
