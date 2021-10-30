### Initial setup of public repository
Create a new empty public repository called `AgML` on the remote and pull it to local.

Create a bare copy of the `AgML-dev` repo. This only clones the main branch, which should be the public release branch.
`$ git clone --bare https://github.com/plant-ai-biophysics-lab/AgML-dev.git`

Change into the `AgML-dev` directory.
`$ cd AgML-dev`

Push the contents of the `AgML-dev` bare duplicate into the new public repo.
`$ git push --mirror https://github.com/plant-ai-biophysics-lab/AgML.git` 

Delete the temporary duplicate of the `AgML-dev` repo. 
`cd ..`
`rm -rf AgML-dev`

### Add a public remote (`AgML`) to the AgML private repo (`AgML-dev`)
`git remote add public https://github.com/plant-ai-biophysics-lab/AgML.git`

After doing this you are able to push branches intended for the public repo from the private repo as follows.
`git push public <branch-name>`

You can then bring changes into your public repo using:
`git fetch public`

### Add a private remote (`AgML-dev`) to the AgML public repo (`AgML`)
`git remote add dev https://github.com/plant-ai-biophysics-lab/AgML-dev.git`

After doing this you are able to push branches intended for the public repo from the private repo as follows.
`git push dev <branch-name>`

You can then bring changes into your public repo using:
`git fetch dev`

### Other useful commands
To list the remote repos that are being tracked:
`git remote -v`
