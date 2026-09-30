//! Teacher-forced scoring through the ordinary full-model forward path.
//! Deliberately independent of sampling and speculative generation.

#[derive(Debug)]
pub struct TokenScore {
    pub position: usize,
    pub token: u32,
    pub nll: f64,
    pub argmax: usize,
}

pub fn token_nll(logits: &[f32], token: u32) -> Result<(f64, usize), String> {
    if logits.is_empty() || token as usize >= logits.len() {
        return Err("empty vocabulary or target token out of range".into());
    }
    if logits.iter().any(|x| !x.is_finite()) {
        return Err("non-finite full-vocabulary logits".into());
    }
    let mut argmax = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[argmax] {
            argmax = i;
        }
    }
    let max = logits[argmax] as f64;
    let normalizer = logits.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>();
    let nll = normalizer.ln() + (max - logits[token as usize] as f64);
    if !nll.is_finite() || nll < 0.0 {
        return Err("invalid NLL".into());
    }
    Ok((nll, argmax))
}

pub fn score<F>(
    tokens: &[u32],
    prefix: usize,
    mut forward: F,
) -> Result<Vec<TokenScore>, String>
where
    F: FnMut(&[u32], usize) -> Result<Vec<f32>, String>,
{
    if prefix == 0 || prefix >= tokens.len() {
        return Err("score prefix must leave nonempty context and continuation".into());
    }
    let mut logits = forward(&tokens[..prefix], 0)?;
    let mut rows = Vec::with_capacity(tokens.len() - prefix);
    for position in prefix..tokens.len() {
        let token = tokens[position];
        let (nll, argmax) = token_nll(&logits, token)?;
        rows.push(TokenScore {
            position,
            token,
            nll,
            argmax,
        });
        if position + 1 < tokens.len() {
            logits = forward(&tokens[position..=position], position)?;
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_normalization_and_shift_invariance() {
        let (a, _) = token_nll(&[0., 1., 2.], 0).unwrap();
        let (b, _) = token_nll(&[10000., 10001., 10002.], 0).unwrap();
        assert!((a - b).abs() < 1e-12);
        assert!((a - (1.0_f64 + 1.0_f64.exp() + 2.0_f64.exp()).ln()).abs() < 1e-12);
        assert_eq!(token_nll(&[0., 0.], 1).unwrap().1, 0);
    }
    #[test]
    fn rejects_corrupt_distribution_and_targets() {
        for values in [
            vec![],
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![f32::NEG_INFINITY],
        ] {
            assert!(token_nll(&values, 0).is_err());
        }
        assert!(token_nll(&[0.], 1).is_err());
    }
    #[test]
    fn teacher_forcing_uses_target_not_prediction_and_correct_positions() {
        let mut calls = Vec::new();
        let rows = score(&[2, 1, 2, 1], 2, |ids, pos| {
            calls.push((ids.to_vec(), pos));
            Ok(vec![5., 1., 0.])
        })
        .unwrap();
        assert_eq!(calls, vec![(vec![2, 1], 0), (vec![2], 2)]);
        assert_eq!(
            rows.iter()
                .map(|r| (r.position, r.token, r.argmax))
                .collect::<Vec<_>>(),
            vec![(2, 2, 0), (3, 1, 0)]
        );
        assert!((rows[0].nll - rows[1].nll - 1.0).abs() < 1e-12);
    }
    #[test]
    fn bounds_and_forward_errors_fail_without_partial_success() {
        for prefix in [0, 2, 3] {
            assert!(
                score(&[0, 1], prefix, |_, _| panic!("must reject before forward"))
                    .is_err()
            );
        }
        assert!(
            score(&[0, 1, 0], 1, |_, pos| if pos == 0 {
                Ok(vec![0., 0.])
            } else {
                Err("device failure".into())
            })
            .is_err()
        );
    }
}
