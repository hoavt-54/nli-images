cat bimpm_img_stack_hypo_only.result | grep -P 'neutral \tneutral' | cut -f3-5 > vbimpm_neutral_neutral.tsv
cat bimpm_baseline.result | grep -f vbimpm_neutral_neutral.tsv | grep -P ' \tcontradiction| \tentailment' > neutral_gs_vbimpm_correct_bimpm_wrong.tsv



#contradiction pairs correctly predicted by bimpm but not by vbimpm
cat bimpm_baseline.result | grep -P 'contradiction \tcontradiction' | cut -f3-5 > bimpm_contradiction_contradiction.tsv
cat bimpm_img_stack_hypo_only.result | grep -f bimpm_contradiction_contradiction.tsv | grep -P ' \tneutral| \tentailment' > contradiction_gs_bimpm_correct_vbimpm_wrong.tsv

#entailment pairs correctly predicted by vbimpm but not by bimpm 
cat bimpm_img_stack_hypo_only.result | grep -P 'entailment \tentailment' | cut -f3-5 > vbimpm_entailment_entailment.tsv 
