# Deployment

So you have written your first pipeline and are ready to put this into a
production environment. This tutorial will help a first time user deploy,
however as operating systems are very different, your mileage might vary.

Also this is not the only method for deploying, and as their doesn't exists a
holy cookbook of deploying. You may find different ways of of doing this. If you
are unsure what a command does write `man $command` and you should get an
explanation what the program does and the options you can use to get different
results

I also want you to understand that the undertaking that your are about to do is
VERY difficult. Involves many steps and isn't something that can be completed in
an afternoon.

## Get a server

The first step is to get the computer that the program should run on. It is a
bad idea to have your pipeline running on your system, as when you turn off your
computer, your pipeline becomes unavailable. Also your user experience with the
computer is going to be worse, since it will be busy processing all the
pictures!

The best option is to contact your local IT department and here what services
they can offer, and if you can, buy a server through them. That way if something
breaks on the server, they can often fix it. They might also provide services
such as backing up or minimal support such as restarting the pipeline.

If your IT department is unable to help you, and this is your first rodeo you
should expect a bump in the budget, but contact sales from
[Lenovo](https://www.lenovo.com), [DELL](https://www.dell.com) or any other
server manufacturer. The Author have good experience with Lenovo, but again
milage might vary.

Requirements for this server should include a GPU, if your pipeline is running
some AI model. Make sure you look through any dependencies and that your server
can handle them. For example if your pipeline includes
[Moose](https://github.com/ENHANCE-PET/MOOSE) you need at least 32 GB of RAM.

Now if you wish to follow this guide, you also need a unix based system, as the
author of this post simply isn't masochistic enough to deploy on windows.

Once you have the server, I will refer this server as production and the
computer you developed on as development.

## The smell of fresh servers

Well your server needs to be on the internet, so grab its IP and ssh to the
server:
`ssh $your-user@$ip-of-the-server`

You need root access to complete this tutorial. You can test that by running
`sudo ls`

If you see your files you have root access. If you get some message about your
user not being in the sudoers file, you need to talk to your IT department about
getting root access. It's important that you mention need the access to install
the pipeline, as root access is not given out lightly.

You should also check that you have internet access, as some servers are unable
access internet for security reasons. To check this run:

`ping google.com`

If your result looks like this:
`From XXX.XXX.XXX.XXX (XXX.XXX.XXX.XXX) icmp_seq=1 Packet filtered`

You need to contact your IT department again and open up the following the
websites:

* https://github.com
* https://files.pythonhosted.org
* https://pipy.org

If your system have a GPU you should also ensure that it works with the command:
`nvidia-smi`

If that doesn't work consult the
[CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

If you need to run docker containers you need some additional software:
(NVIDIA Container Toolkit)[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html]

Note that installation of nvidia software is something most IT department can do
for you, and you should let them if you're inexperienced as an incorrect CUDA
installation can brick your system, either now or later when you try and update.

### Getting your program ready for production

The next step is to recreate your development environment. I.E getting your
program to run on the server. If you need to transport files such as model
weights you can do it using the following command running from your development
computer:

`scp $models_wights_file $your-user@$ip-of-the-server:/home/$your-user`

You might encounter that the your program does't work because you have hardcoded
some paths that doesn't exists. Edit your pipeline such that these path come
from environment variables. Here is a little code snippet how to do it:

```python
# old code:
path = "/path/to/something"

# new code
from os import environ

ENVIRONMENT_VARIABLE_NAME = "MY_PIPELINE_{pick a good name}"
ENVIRONMENT_VARIABLE_VALUE = environ.get(ENVIRONMENT_VARIABLE_NAME,
                                         "/path/to/something")
```

You might need to do further changes to your program, if so, read the error
messages and fix the errors!

## Footwork for the service

So far the program have either been running as root or as your user. This is
undesirable because either:

* Root - has too many rights and pose a security risk.
* Your user - is bound to your personality and might cause some privacy problems
as from the system site it looks like you have personally looked through each
study passing through the pipeline. If your account become inactive,
various services stop running which is also bad. On some systems you also have
more rights than a system user, which again is undesirable.

So lets create a user for the pipeline:
`sudo useradd pipeline --comment "The user running the pipeline" --system --shell /sbin/nologin`

The `--system` makes it clear to the operating system this not a person, but a
program. The `--shell /sbin/nologin` makes it so if your dicomnode becomes
compromised, it becomes more difficult to compromise the entire system.

Note that this might break your program, if your program needs a shell to
execute. First of all \Shame.gif\ on you and your cow, but you can change this
by:
`sudo chsh pipeline /bin/bash`

And just live with the security vulnerability.

Next you want to add a pipeline group. This make it so you can edit the files
with out root access

`sudo groupadd pipelineadmin`

and add yourself to the created group

`sudo usermod -aG pipelineadmin $your-user`

Next create the folder where the program should run from as a service. No your
home directory isn't good enough. You might have some restrictions where you
should place this, but for the rest of this guide it assumes it's in
`/opt/pipeline` you can replace this with other destinations.

`sudo mkdir /opt/pipeline`

Take your pipeline installation and move it using the `mv` command inside
`/opt/pipeline`. If you get access denied message put a `sudo` in front of the
command.

Now we change the ownership such that you both you and the pipeline have the
right accesses.

`sudo chown -R pipeline:pipelineadmin /opt/pipeline`

Note that if you later add files to your pipeline it's important that rerun this
command, otherwise your pipeline might fail, when it tries to open files.

## Setting up the service

Okay we are finally ready to tell the operating system, that it should run the
pipeline. This is done through a system called `systemd` and managed through
the `systemctl` command.

This is done by creating a file in folder `/lib/system/systemd`
For example: `sudo nano /lib/system/systemd/pipeline.service`

The filename is important, as it is what systemctl uses as command destination.
So if the filename was `pipeline123.service` you can control it using:
`systemctl <command> pipeline123`

You want to file to contain:
```
[Unit]
Description=Post processing pipeline for of medical images
After=network.target

[Service]
User=pipeline
Group=pipeline
WorkingDirectory=/opt/pipeline
ExecStart=/opt/pipeline/venv/bin/python /opt/pipeline/script.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Lets go through each line: The Unit group describes information about the
service in relation to other services. The description is there to help other
understand what the program does. After tells the operating system that this
program should be executed after the network service have started. If you don't
have this line you'll get some funny errors when you try to open a server.

The Service label describes the actual programs that the operating system should
run.
In this case we specify that it should be process should be run by the pipeline
user and pipeline group. The working directory should be the `/opt/pipeline`
Which is where relative path will be calculated from.

When describing path you should ALWAYS use absolute paths. An absolute path
starts with a '/' character and gets the file from the root of the filesystem.

ExecStart describes the program that should run. Now it's very important that
you use the python inside of the virtual environment, otherwise you'll get the
system python which have different packages.

And if you're not using virtual environment: /shame.gif/ to you and your cow.

Restart is a directive how the system should handle restarting your program, if
it stops. on-failure means that if the program exits and have non 0 exits code
then the system will attempt to restart the program.

You can check exits codes using: `echo $?` command.

Finally install how describes how the system should install the service.
Systemd have some managers that is responsible for executing the services on the
system. If you wanna learn more:
[The Rabbit hole](https://www.baeldung.com/linux/systemd-target-multi-user)

Otherwise just set it to multi-user.target and continue to live happily in
ignorance.

If you actually wanna get good at writing these files I recommend
(This wiki page)[https://fedoraproject.org/wiki/Packaging:Systemd]
Note again your milage might vary on operating to operating system, however if
it stands in this guide i would assume (and be wrong sometimes) on any Unix os

Afterwards run

`sudo systemctl daemon-reload` to make the system recognize that there's a new
service

`sudo systemctl start pipeline` to make the system start the system

And congratulations the pipeline is installed!

## Death, Taxes and failing systemd processes

So you followed the guide to perfection, didn't miss a single slash and even
read the entire guide, twice! But something does't work. Well let me tell you,
I lied, the installation process is far from over!

`systemctl status pipeline` to check on the status of the pipeline, there should
be an `Active: active (running)` or an `Active: failure(exited)` line in the
program output. This will tell you if the program or not.

To restart the service you can run `systemctl restart pipeline` to force the
system to restart your program, however you should probably fix, what is wrong
first.

As to what could have gone wrong, you'll have to read log files and figure out
yourself. I'll list some common pitfalls, however first things first, getting
the logs.

To get the logs, you can run:
`journalctl -ru pipeline` or `journalctl -o verbose -ru pipeline` depending on
how screwed you think you are.

Either way this is where google skills needs to be sharp.

### Ports

If you remember from your pipeline class, you most likely define a port that the
server should open on or if you don't it defaults to port 104, which is
privileged as all ports below 1024. Privileged ports can only opened by root,
which the pipeline system user isn't.

To fix this, either move an unprivileged ports such as 11112, set the user to
root (And hear the tolls of shame!), use authbind,
(Read this tutorial)[https://www.baeldung.com/linux/bind-process-privileged-port],
or wait for me (or contribute) to implement some fancy code and update this
 tutorial.

### Firewall

Most systems have a firewall that prevents message from the outside reaching the
service process.

`sudo firewall-cmd --permanent --add-port=104/tcp`
`sudo firewall-cmd --reload`

Here I have assume you use port 104, change it to whatever port you are using.

Now it might not be the system that's acting as a firewall, sometimes your it
department does packet filtering on the network. In that case you need to get in
touch with your IT department and then get allow them to allow your network
traffic.

This will likely be the case if you also needed to get the github/pythonhosted
urls opened by your IT Department

### Selinux

If your system is running Selinux, you need to allow process to all that it
needs. Sadly my SelinuxFoo is insufficient for me to write a good guide on how
to set good security contexts.

Look at /var/log/audit/audit.log

## Tips, Tricks and other good stuff

So at this point you hopefully have your pipeline up and running. These are some
tips that might ease some pain points.

### Configuration

So obviously it's bad to have passwords lying in code that is available on the
internet or you remember those paths, that you made environment variables for.

I have found 2 ways of dealing with this issue on:

#### Config files
So first i create an python file with a bunch of empty definitions for instance:
```python
#config.py
DATABASE_PASSWORD = ""
SOME_PATH_I_NEED = ""
# etc etc etc
```

Then the production system i create a branch called `local` using
`git branch local` and modify and commit the config file.

The way to update the system then becomes:
```bash
# At the development
git commit -m "bla bla bla" -S
git push

# At production server
git checkout master
git pull
git checkout local
git merge master

sudo systemctl restart pipeline
```

And yes I suck at git.

#### Environment variables

So if you remember from earlier in the guide I suggested you used environment
variables. You can set them using:

`systemctl edit pipeline`

You'll get a editor, where you can type:

[Service]
Environment="SECRET=pGNqduRFkB4K9C2vijOmUDa2kPtUhArN"
Environment="ENVIRONMENT_VARIABLE_NAME=ENVIRONMENT_VARIABLE_VALUE"



