//! Exact/unknown/inapplicable and interval facts match the Features-4 contract.
use super::*;
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measure {
    pub value: f64,
    pub known: bool,
    pub applicable: bool,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

impl Measure {
    pub const fn absent() -> Self {
        Self {
            value: 0.,
            known: false,
            applicable: false,
            lower: None,
            upper: None,
        }
    }
    pub const fn unknown() -> Self {
        Self {
            applicable: true,
            ..Self::absent()
        }
    }
    pub fn exact(value: impl Into<f64>) -> Self {
        let value = value.into();
        Self {
            value,
            known: true,
            applicable: true,
            lower: Some(value),
            upper: Some(value),
        }
    }
    pub fn boolean(value: bool) -> Self {
        Self::exact(u8::from(value))
    }
    pub fn interval(lower: Option<f64>, upper: Option<f64>) -> Self {
        if let (Some(l), Some(u)) = (lower, upper) {
            if l == u {
                return Self::exact(l);
            }
        }
        Self {
            lower,
            upper,
            ..Self::unknown()
        }
    }
    pub fn emit(self, selection: &layout::Selection, out: &mut Vec<f32>) -> Result<(), CoreError> {
        check(
            self.value.is_finite()
                && self.lower.is_none_or(f64::is_finite)
                && self.upper.is_none_or(f64::is_finite)
                && !(self.lower.zip(self.upper).is_some_and(|(l, u)| l > u)),
            "invalid derived bounds",
        )?;
        if selection.width == 3 {
            check(
                self.known || self.lower.is_none() && self.upper.is_none(),
                "compact feature would discard bounds",
            )?;
        }
        out.extend_from_slice(&[
            (self.value / selection.scale) as f32,
            f32::from(self.known),
            f32::from(self.applicable),
        ]);
        if selection.width == 7 {
            out.extend_from_slice(&[
                (self.lower.unwrap_or(0.) / selection.scale) as f32,
                f32::from(self.lower.is_some()),
                (self.upper.unwrap_or(0.) / selection.scale) as f32,
                f32::from(self.upper.is_some()),
            ]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_unknowns_and_applicability_have_distinct_encodings() {
        let full = layout::Selection {
            index: 0,
            scale: 10.,
            width: 7,
        };
        let mut values = Vec::new();
        for m in [
            Measure::exact(5),
            Measure::unknown(),
            Measure::absent(),
            Measure::interval(Some(2.), None),
        ] {
            m.emit(&full, &mut values).unwrap();
        }
        assert_eq!(
            values,
            [
                0.5, 1., 1., 0.5, 1., 0.5, 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 1., 0.2, 1., 0., 0.
            ]
        );
        assert_eq!(Measure::interval(Some(3.), Some(3.)), Measure::exact(3));
        assert!(Measure::interval(Some(3.), Some(2.))
            .emit(&full, &mut values)
            .is_err());
        assert!(Measure::exact(f64::INFINITY)
            .emit(&full, &mut values)
            .is_err());
        let compact = layout::Selection { width: 3, ..full };
        assert!(Measure::interval(Some(2.), None)
            .emit(&compact, &mut values)
            .is_err());
    }
}
