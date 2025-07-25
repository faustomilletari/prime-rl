# from environments/





# from verifiers github
VF_COMMIT="ce362d4"
uv pip install 'wordle @ git+https://github.com/willccbb/verifiers.git@${VF_COMMIT}#subdirectory=environments/wordle'
uv pip install 'simpleqa @ git+https://github.com/willccbb/verifiers.git@${VF_COMMIT}#subdirectory=environments/simpleqa'



