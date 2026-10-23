### Why these decisions were made

| Decision | Reason |
|---|---|
| pYIN supplies downstream notes | It currently produces absolute pitch and note timing; pilot classifiers produce only rest/pitch class. |
| Keep CNN/hybrid visible as experiments | Their measured score-class results have not improved on the baseline. |
| No automatic label corrections | Listening review and timing verification are still incomplete. |
| Operator supplies key and beats | Validated learned key/beat heads are not implemented. |
| Keep original time origin | Downstream chord/audio plots must refer to the same recording. |
| Preserve all source recordings | Research results must remain reproducible and original evidence unchanged. |
