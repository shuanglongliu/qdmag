# sed -i --follow-symlinks "s/T per ms/T per ps/g; s/rad ms/rad ps/g; s/omega ms/omega ps/g" *.py
# sed -i --follow-symlinks "/imbalance/d" *.yaml
# sed -i --follow-symlinks "/multiphonon/d" *.yaml
# sed -i --follow-symlinks "s/dynamics, n_thread/dynamics, states, n_thread/" *.py
sed -i --follow-symlinks "s/get_h_Zeeman_Mv_tot/get_h_Zeeman_Mv_eff/" *.py
