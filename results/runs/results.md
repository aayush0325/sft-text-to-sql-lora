| model | r | alpha | target_modules | trainable_params | final_train_loss | eval_loss | execution_accuracy | valid_sql_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | - | - | - | - | - | - | 0.5000 | 0.8438 |
| attn_r8 | 8 | 16 | q_proj+k_proj+v_proj+o_proj | 1703936 | 0.2188 | 0.2181 | 0.6875 | 0.9062 |
| attn_mlp_r16 | 16 | 32 | q_proj+k_proj+v_proj+o_proj+gate_proj+up_proj+down_proj | 11272192 | 0.1944 | 0.1919 | 0.7188 | 0.9375 |
| mlp_r32 | 32 | 64 | gate_proj+up_proj+down_proj | 15728640 | 0.1927 | 0.1890 | 0.8438 | 0.9688 |
| attn_r64_rslora | 64 | 32 | q_proj+k_proj+v_proj+o_proj | 13631488 | 0.1941 | 0.1944 | 0.7812 | 0.9062 |
| all_linear_r16_dora | 16 | 32 | q_proj+k_proj+v_proj+o_proj+gate_proj+up_proj+down_proj | 11649024 | 0.1940 | 0.1915 | 0.7812 | 0.9375 |
