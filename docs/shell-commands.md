# Shell mode: commands checklist

| # | Command | Status |
| - | ------- | ------ |
| 1 | `ls` | working |
| 2 | `pwd` | working |
| 3 | `cat somefile.txt` | working |
| 4 | `mkdir subdir` | working |
| 5 | `python3 --version` | working |
| 6 | `git status` | working |
| 7 | `rm somefile.txt` | working |
| 8 | `sleep 15` | working |
| 9 | (empty input) | working |
| 10 | `not_a_real_command_xyz` | working |
| 11 | `cat /etc/passwd` | blocked |
| 12 | `cat ../secret.txt` | blocked |
| 13 | `ls foo \| grep foo` | not working |
| 14 | `echo hi > out.txt` | not working |
| 15 | `cat *.txt` | not working |
| 16 | `echo $HOME` | not working |
| 17 | `cd subdir && ls` | not working |
| 18 | `ls ~` | not working |
