ич!
┼Ц
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68╖Б 
|
salt_pred/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namesalt_pred/kernel
u
$salt_pred/kernel/Read/ReadVariableOpReadVariableOpsalt_pred/kernel*
_output_shapes

:@*
dtype0
t
salt_pred/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namesalt_pred/bias
m
"salt_pred/bias/Read/ReadVariableOpReadVariableOpsalt_pred/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
Х
salt_seq/lstm_cell_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*-
shared_namesalt_seq/lstm_cell_30/kernel
О
0salt_seq/lstm_cell_30/kernel/Read/ReadVariableOpReadVariableOpsalt_seq/lstm_cell_30/kernel*
_output_shapes
:	А*
dtype0
й
&salt_seq/lstm_cell_30/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*7
shared_name(&salt_seq/lstm_cell_30/recurrent_kernel
в
:salt_seq/lstm_cell_30/recurrent_kernel/Read/ReadVariableOpReadVariableOp&salt_seq/lstm_cell_30/recurrent_kernel*
_output_shapes
:	 А*
dtype0
Н
salt_seq/lstm_cell_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namesalt_seq/lstm_cell_30/bias
Ж
.salt_seq/lstm_cell_30/bias/Read/ReadVariableOpReadVariableOpsalt_seq/lstm_cell_30/bias*
_output_shapes	
:А*
dtype0
У
qty_seq/lstm_cell_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*,
shared_nameqty_seq/lstm_cell_31/kernel
М
/qty_seq/lstm_cell_31/kernel/Read/ReadVariableOpReadVariableOpqty_seq/lstm_cell_31/kernel*
_output_shapes
:	А*
dtype0
з
%qty_seq/lstm_cell_31/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*6
shared_name'%qty_seq/lstm_cell_31/recurrent_kernel
а
9qty_seq/lstm_cell_31/recurrent_kernel/Read/ReadVariableOpReadVariableOp%qty_seq/lstm_cell_31/recurrent_kernel*
_output_shapes
:	 А*
dtype0
Л
qty_seq/lstm_cell_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_nameqty_seq/lstm_cell_31/bias
Д
-qty_seq/lstm_cell_31/bias/Read/ReadVariableOpReadVariableOpqty_seq/lstm_cell_31/bias*
_output_shapes	
:А*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
К
Adam/salt_pred/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/salt_pred/kernel/m
Г
+Adam/salt_pred/kernel/m/Read/ReadVariableOpReadVariableOpAdam/salt_pred/kernel/m*
_output_shapes

:@*
dtype0
В
Adam/salt_pred/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/salt_pred/bias/m
{
)Adam/salt_pred/bias/m/Read/ReadVariableOpReadVariableOpAdam/salt_pred/bias/m*
_output_shapes
:*
dtype0
г
#Adam/salt_seq/lstm_cell_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*4
shared_name%#Adam/salt_seq/lstm_cell_30/kernel/m
Ь
7Adam/salt_seq/lstm_cell_30/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/salt_seq/lstm_cell_30/kernel/m*
_output_shapes
:	А*
dtype0
╖
-Adam/salt_seq/lstm_cell_30/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*>
shared_name/-Adam/salt_seq/lstm_cell_30/recurrent_kernel/m
░
AAdam/salt_seq/lstm_cell_30/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/salt_seq/lstm_cell_30/recurrent_kernel/m*
_output_shapes
:	 А*
dtype0
Ы
!Adam/salt_seq/lstm_cell_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/salt_seq/lstm_cell_30/bias/m
Ф
5Adam/salt_seq/lstm_cell_30/bias/m/Read/ReadVariableOpReadVariableOp!Adam/salt_seq/lstm_cell_30/bias/m*
_output_shapes	
:А*
dtype0
б
"Adam/qty_seq/lstm_cell_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*3
shared_name$"Adam/qty_seq/lstm_cell_31/kernel/m
Ъ
6Adam/qty_seq/lstm_cell_31/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/qty_seq/lstm_cell_31/kernel/m*
_output_shapes
:	А*
dtype0
╡
,Adam/qty_seq/lstm_cell_31/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*=
shared_name.,Adam/qty_seq/lstm_cell_31/recurrent_kernel/m
о
@Adam/qty_seq/lstm_cell_31/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/qty_seq/lstm_cell_31/recurrent_kernel/m*
_output_shapes
:	 А*
dtype0
Щ
 Adam/qty_seq/lstm_cell_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/qty_seq/lstm_cell_31/bias/m
Т
4Adam/qty_seq/lstm_cell_31/bias/m/Read/ReadVariableOpReadVariableOp Adam/qty_seq/lstm_cell_31/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/salt_pred/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/salt_pred/kernel/v
Г
+Adam/salt_pred/kernel/v/Read/ReadVariableOpReadVariableOpAdam/salt_pred/kernel/v*
_output_shapes

:@*
dtype0
В
Adam/salt_pred/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/salt_pred/bias/v
{
)Adam/salt_pred/bias/v/Read/ReadVariableOpReadVariableOpAdam/salt_pred/bias/v*
_output_shapes
:*
dtype0
г
#Adam/salt_seq/lstm_cell_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*4
shared_name%#Adam/salt_seq/lstm_cell_30/kernel/v
Ь
7Adam/salt_seq/lstm_cell_30/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/salt_seq/lstm_cell_30/kernel/v*
_output_shapes
:	А*
dtype0
╖
-Adam/salt_seq/lstm_cell_30/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*>
shared_name/-Adam/salt_seq/lstm_cell_30/recurrent_kernel/v
░
AAdam/salt_seq/lstm_cell_30/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/salt_seq/lstm_cell_30/recurrent_kernel/v*
_output_shapes
:	 А*
dtype0
Ы
!Adam/salt_seq/lstm_cell_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/salt_seq/lstm_cell_30/bias/v
Ф
5Adam/salt_seq/lstm_cell_30/bias/v/Read/ReadVariableOpReadVariableOp!Adam/salt_seq/lstm_cell_30/bias/v*
_output_shapes	
:А*
dtype0
б
"Adam/qty_seq/lstm_cell_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*3
shared_name$"Adam/qty_seq/lstm_cell_31/kernel/v
Ъ
6Adam/qty_seq/lstm_cell_31/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/qty_seq/lstm_cell_31/kernel/v*
_output_shapes
:	А*
dtype0
╡
,Adam/qty_seq/lstm_cell_31/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*=
shared_name.,Adam/qty_seq/lstm_cell_31/recurrent_kernel/v
о
@Adam/qty_seq/lstm_cell_31/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/qty_seq/lstm_cell_31/recurrent_kernel/v*
_output_shapes
:	 А*
dtype0
Щ
 Adam/qty_seq/lstm_cell_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/qty_seq/lstm_cell_31/bias/v
Т
4Adam/qty_seq/lstm_cell_31/bias/v/Read/ReadVariableOpReadVariableOp Adam/qty_seq/lstm_cell_31/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
∙G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*┤G
valueкGBзG BаG
В
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
┴
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
┴
cell

state_spec
	variables
trainable_variables
regularization_losses
 	keras_api
!_random_generator
"__call__
*#&call_and_return_all_conditional_losses*
е
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(_random_generator
)__call__
**&call_and_return_all_conditional_losses* 
е
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses* 
О
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 
ж

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
ф
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate8mР9mСEmТFmУGmФHmХImЦJmЧ8vШ9vЩEvЪFvЫGvЬHvЭIvЮJvЯ*
<
E0
F1
G2
H3
I4
J5
86
97*
<
E0
F1
G2
H3
I4
J5
86
97*
* 
░
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Pserving_default* 
у
Q
state_size

Ekernel
Frecurrent_kernel
Gbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V_random_generator
W__call__
*X&call_and_return_all_conditional_losses*
* 

E0
F1
G2*

E0
F1
G2*
* 
Я

Ystates
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
у
_
state_size

Hkernel
Irecurrent_kernel
Jbias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d_random_generator
e__call__
*f&call_and_return_all_conditional_losses*
* 

H0
I1
J2*

H0
I1
J2*
* 
Я

gstates
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
С
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
$	variables
%trainable_variables
&regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
С
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
С
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEsalt_pred/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEsalt_pred/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 
Ф
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
Аlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEsalt_seq/lstm_cell_30/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&salt_seq/lstm_cell_30/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEsalt_seq/lstm_cell_30/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEqty_seq/lstm_cell_31/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%qty_seq/lstm_cell_31/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEqty_seq/lstm_cell_31/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*

Б0*
* 
* 
* 
* 

E0
F1
G2*

E0
F1
G2*
* 
Ш
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 

H0
I1
J2*

H0
I1
J2*
* 
Ш
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
`	variables
atrainable_variables
bregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

Мtotal

Нcount
О	variables
П	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

М0
Н1*

О	variables*
Г}
VARIABLE_VALUEAdam/salt_pred/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/salt_pred/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/salt_seq/lstm_cell_30/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUE-Adam/salt_seq/lstm_cell_30/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/salt_seq/lstm_cell_30/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/qty_seq/lstm_cell_31/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/qty_seq/lstm_cell_31/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/qty_seq/lstm_cell_31/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/salt_pred/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/salt_pred/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/salt_seq/lstm_cell_30/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUE-Adam/salt_seq/lstm_cell_30/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/salt_seq/lstm_cell_30/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/qty_seq/lstm_cell_31/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUE,Adam/qty_seq/lstm_cell_31/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/qty_seq/lstm_cell_31/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
И
serving_default_quantity_dataPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
Д
serving_default_salt_dataPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
╟
StatefulPartitionedCallStatefulPartitionedCallserving_default_quantity_dataserving_default_salt_dataqty_seq/lstm_cell_31/kernel%qty_seq/lstm_cell_31/recurrent_kernelqty_seq/lstm_cell_31/biassalt_seq/lstm_cell_30/kernel&salt_seq/lstm_cell_30/recurrent_kernelsalt_seq/lstm_cell_30/biassalt_pred/kernelsalt_pred/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_127830
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Т
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$salt_pred/kernel/Read/ReadVariableOp"salt_pred/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp0salt_seq/lstm_cell_30/kernel/Read/ReadVariableOp:salt_seq/lstm_cell_30/recurrent_kernel/Read/ReadVariableOp.salt_seq/lstm_cell_30/bias/Read/ReadVariableOp/qty_seq/lstm_cell_31/kernel/Read/ReadVariableOp9qty_seq/lstm_cell_31/recurrent_kernel/Read/ReadVariableOp-qty_seq/lstm_cell_31/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/salt_pred/kernel/m/Read/ReadVariableOp)Adam/salt_pred/bias/m/Read/ReadVariableOp7Adam/salt_seq/lstm_cell_30/kernel/m/Read/ReadVariableOpAAdam/salt_seq/lstm_cell_30/recurrent_kernel/m/Read/ReadVariableOp5Adam/salt_seq/lstm_cell_30/bias/m/Read/ReadVariableOp6Adam/qty_seq/lstm_cell_31/kernel/m/Read/ReadVariableOp@Adam/qty_seq/lstm_cell_31/recurrent_kernel/m/Read/ReadVariableOp4Adam/qty_seq/lstm_cell_31/bias/m/Read/ReadVariableOp+Adam/salt_pred/kernel/v/Read/ReadVariableOp)Adam/salt_pred/bias/v/Read/ReadVariableOp7Adam/salt_seq/lstm_cell_30/kernel/v/Read/ReadVariableOpAAdam/salt_seq/lstm_cell_30/recurrent_kernel/v/Read/ReadVariableOp5Adam/salt_seq/lstm_cell_30/bias/v/Read/ReadVariableOp6Adam/qty_seq/lstm_cell_31/kernel/v/Read/ReadVariableOp@Adam/qty_seq/lstm_cell_31/recurrent_kernel/v/Read/ReadVariableOp4Adam/qty_seq/lstm_cell_31/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__traced_save_129461
б	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesalt_pred/kernelsalt_pred/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratesalt_seq/lstm_cell_30/kernel&salt_seq/lstm_cell_30/recurrent_kernelsalt_seq/lstm_cell_30/biasqty_seq/lstm_cell_31/kernel%qty_seq/lstm_cell_31/recurrent_kernelqty_seq/lstm_cell_31/biastotalcountAdam/salt_pred/kernel/mAdam/salt_pred/bias/m#Adam/salt_seq/lstm_cell_30/kernel/m-Adam/salt_seq/lstm_cell_30/recurrent_kernel/m!Adam/salt_seq/lstm_cell_30/bias/m"Adam/qty_seq/lstm_cell_31/kernel/m,Adam/qty_seq/lstm_cell_31/recurrent_kernel/m Adam/qty_seq/lstm_cell_31/bias/mAdam/salt_pred/kernel/vAdam/salt_pred/bias/v#Adam/salt_seq/lstm_cell_30/kernel/v-Adam/salt_seq/lstm_cell_30/recurrent_kernel/v!Adam/salt_seq/lstm_cell_30/bias/v"Adam/qty_seq/lstm_cell_31/kernel/v,Adam/qty_seq/lstm_cell_31/recurrent_kernel/v Adam/qty_seq/lstm_cell_31/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__traced_restore_129564Г╓
╟

╙
$__inference_signature_wrapper_127830
quantity_data
	salt_data
unknown:	А
	unknown_0:	 А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	 А
	unknown_4:	А
	unknown_5:@
	unknown_6:
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCall	salt_dataquantity_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_125548o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namequantity_data:VR
+
_output_shapes
:         
#
_user_specified_name	salt_data
О8
Е
C__inference_qty_seq_layer_call_and_return_conditional_losses_126239

inputs&
lstm_cell_31_126157:	А&
lstm_cell_31_126159:	 А"
lstm_cell_31_126161:	А
identityИв$lstm_cell_31/StatefulPartitionedCallвwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskї
$lstm_cell_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_31_126157lstm_cell_31_126159lstm_cell_31_126161*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_126111n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╖
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_31_126157lstm_cell_31_126159lstm_cell_31_126161*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_126170*
condR
while_cond_126169*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          u
NoOpNoOp%^lstm_cell_31/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_31/StatefulPartitionedCall$lstm_cell_31/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╡
├
while_cond_126750
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_126750___redundant_placeholder04
0while_while_cond_126750___redundant_placeholder14
0while_while_cond_126750___redundant_placeholder24
0while_while_cond_126750___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
П8
Ж
D__inference_salt_seq_layer_call_and_return_conditional_losses_125889

inputs&
lstm_cell_30_125807:	А&
lstm_cell_30_125809:	 А"
lstm_cell_30_125811:	А
identityИв$lstm_cell_30/StatefulPartitionedCallвwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskї
$lstm_cell_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_30_125807lstm_cell_30_125809lstm_cell_30_125811*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_125761n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╖
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_30_125807lstm_cell_30_125809lstm_cell_30_125811*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_125820*
condR
while_cond_125819*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          u
NoOpNoOp%^lstm_cell_30/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_30/StatefulPartitionedCall$lstm_cell_30/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
∙"
у
while_body_125629
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_30_125653_0:	А.
while_lstm_cell_30_125655_0:	 А*
while_lstm_cell_30_125657_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_30_125653:	А,
while_lstm_cell_30_125655:	 А(
while_lstm_cell_30_125657:	АИв*while/lstm_cell_30/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0│
*while/lstm_cell_30/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_30_125653_0while_lstm_cell_30_125655_0while_lstm_cell_30_125657_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_125615▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_30/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥Р
while/Identity_4Identity3while/lstm_cell_30/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          Р
while/Identity_5Identity3while/lstm_cell_30/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          y

while/NoOpNoOp+^while/lstm_cell_30/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_30_125653while_lstm_cell_30_125653_0"8
while_lstm_cell_30_125655while_lstm_cell_30_125655_0"8
while_lstm_cell_30_125657while_lstm_cell_30_125657_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2X
*while/lstm_cell_30/StatefulPartitionedCall*while/lstm_cell_30/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
йJ
Ь
D__inference_salt_seq_layer_call_and_return_conditional_losses_126835

inputs>
+lstm_cell_30_matmul_readvariableop_resource:	А@
-lstm_cell_30_matmul_1_readvariableop_resource:	 А;
,lstm_cell_30_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_30/BiasAdd/ReadVariableOpв"lstm_cell_30/MatMul/ReadVariableOpв$lstm_cell_30/MatMul_1/ReadVariableOpвwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskП
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АМ
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АН
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А^
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▌
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitn
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*'
_output_shapes
:          p
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*'
_output_shapes
:          w
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          h
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*'
_output_shapes
:          Ж
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          {
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          p
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*'
_output_shapes
:          e
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          К
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_126751*
condR
while_cond_126750*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          └
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╪
c
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_129104

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:          [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╫

╧
&__inference_model_layer_call_fn_127206
inputs_0
inputs_1
unknown:	А
	unknown_0:	 А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	 А
	unknown_4:	А
	unknown_5:@
	unknown_6:
identityИвStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_127061o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:         
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/1
╫

╧
&__inference_model_layer_call_fn_127184
inputs_0
inputs_1
unknown:	А
	unknown_0:	 А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	 А
	unknown_4:	А
	unknown_5:@
	unknown_6:
identityИвStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_126597o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:         
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/1
Я8
╨
while_body_128835
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_31_matmul_readvariableop_resource_0:	АH
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:	 АC
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_31_matmul_readvariableop_resource:	АF
3while_lstm_cell_31_matmul_1_readvariableop_resource:	 АA
2while_lstm_cell_31_biasadd_readvariableop_resource:	АИв)while/lstm_cell_31/BiasAdd/ReadVariableOpв(while/lstm_cell_31/MatMul/ReadVariableOpв*while/lstm_cell_31/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Э
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0║
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0б
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АЫ
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0з
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitz
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*'
_output_shapes
:          |
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*'
_output_shapes
:          Ж
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          t
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ш
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          Н
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          |
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*'
_output_shapes
:          q
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          Ь
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥y
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          y
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          ╨

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
П
╖
(__inference_qty_seq_layer_call_fn_128457
inputs_0
unknown:	А
	unknown_0:	 А
	unknown_1:	А
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_126048o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
Я8
╨
while_body_128219
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_30_matmul_readvariableop_resource_0:	АH
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:	 АC
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_30_matmul_readvariableop_resource:	АF
3while_lstm_cell_30_matmul_1_readvariableop_resource:	 АA
2while_lstm_cell_30_biasadd_readvariableop_resource:	АИв)while/lstm_cell_30/BiasAdd/ReadVariableOpв(while/lstm_cell_30/MatMul/ReadVariableOpв*while/lstm_cell_30/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Э
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0║
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0б
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АЫ
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0з
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitz
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*'
_output_shapes
:          |
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*'
_output_shapes
:          Ж
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          t
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*'
_output_shapes
:          Ш
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          Н
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          |
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*'
_output_shapes
:          q
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          Ь
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥y
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          y
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          ╨

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
╡
├
while_cond_128548
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_128548___redundant_placeholder04
0while_while_cond_128548___redundant_placeholder14
0while_while_cond_128548___redundant_placeholder24
0while_while_cond_128548___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╡
├
while_cond_128834
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_128834___redundant_placeholder04
0while_while_cond_128834___redundant_placeholder14
0while_while_cond_128834___redundant_placeholder24
0while_while_cond_128834___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╡
├
while_cond_125628
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_125628___redundant_placeholder04
0while_while_cond_125628___redundant_placeholder14
0while_while_cond_125628___redundant_placeholder24
0while_while_cond_125628___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
Ц

у
qty_seq_while_cond_127558,
(qty_seq_while_qty_seq_while_loop_counter2
.qty_seq_while_qty_seq_while_maximum_iterations
qty_seq_while_placeholder
qty_seq_while_placeholder_1
qty_seq_while_placeholder_2
qty_seq_while_placeholder_3.
*qty_seq_while_less_qty_seq_strided_slice_1D
@qty_seq_while_qty_seq_while_cond_127558___redundant_placeholder0D
@qty_seq_while_qty_seq_while_cond_127558___redundant_placeholder1D
@qty_seq_while_qty_seq_while_cond_127558___redundant_placeholder2D
@qty_seq_while_qty_seq_while_cond_127558___redundant_placeholder3
qty_seq_while_identity
В
qty_seq/while/LessLessqty_seq_while_placeholder*qty_seq_while_less_qty_seq_strided_slice_1*
T0*
_output_shapes
: [
qty_seq/while/IdentityIdentityqty_seq/while/Less:z:0*
T0
*
_output_shapes
: "9
qty_seq_while_identityqty_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╡
├
while_cond_128977
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_128977___redundant_placeholder04
0while_while_cond_128977___redundant_placeholder14
0while_while_cond_128977___redundant_placeholder24
0while_while_cond_128977___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╛
█
model_qty_seq_while_cond_1253148
4model_qty_seq_while_model_qty_seq_while_loop_counter>
:model_qty_seq_while_model_qty_seq_while_maximum_iterations#
model_qty_seq_while_placeholder%
!model_qty_seq_while_placeholder_1%
!model_qty_seq_while_placeholder_2%
!model_qty_seq_while_placeholder_3:
6model_qty_seq_while_less_model_qty_seq_strided_slice_1P
Lmodel_qty_seq_while_model_qty_seq_while_cond_125314___redundant_placeholder0P
Lmodel_qty_seq_while_model_qty_seq_while_cond_125314___redundant_placeholder1P
Lmodel_qty_seq_while_model_qty_seq_while_cond_125314___redundant_placeholder2P
Lmodel_qty_seq_while_model_qty_seq_while_cond_125314___redundant_placeholder3 
model_qty_seq_while_identity
Ъ
model/qty_seq/while/LessLessmodel_qty_seq_while_placeholder6model_qty_seq_while_less_model_qty_seq_strided_slice_1*
T0*
_output_shapes
: g
model/qty_seq/while/IdentityIdentitymodel/qty_seq/while/Less:z:0*
T0
*
_output_shapes
: "E
model_qty_seq_while_identity%model/qty_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
б
G
+__inference_salt_seq_2_layer_call_fn_129067

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_126562`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
тH
Р
model_qty_seq_while_body_1253158
4model_qty_seq_while_model_qty_seq_while_loop_counter>
:model_qty_seq_while_model_qty_seq_while_maximum_iterations#
model_qty_seq_while_placeholder%
!model_qty_seq_while_placeholder_1%
!model_qty_seq_while_placeholder_2%
!model_qty_seq_while_placeholder_37
3model_qty_seq_while_model_qty_seq_strided_slice_1_0s
omodel_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_qty_seq_tensorarrayunstack_tensorlistfromtensor_0T
Amodel_qty_seq_while_lstm_cell_31_matmul_readvariableop_resource_0:	АV
Cmodel_qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource_0:	 АQ
Bmodel_qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource_0:	А 
model_qty_seq_while_identity"
model_qty_seq_while_identity_1"
model_qty_seq_while_identity_2"
model_qty_seq_while_identity_3"
model_qty_seq_while_identity_4"
model_qty_seq_while_identity_55
1model_qty_seq_while_model_qty_seq_strided_slice_1q
mmodel_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_qty_seq_tensorarrayunstack_tensorlistfromtensorR
?model_qty_seq_while_lstm_cell_31_matmul_readvariableop_resource:	АT
Amodel_qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource:	 АO
@model_qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource:	АИв7model/qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOpв6model/qty_seq/while/lstm_cell_31/MatMul/ReadVariableOpв8model/qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOpЦ
Emodel/qty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ь
7model/qty_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemomodel_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_qty_seq_tensorarrayunstack_tensorlistfromtensor_0model_qty_seq_while_placeholderNmodel/qty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0╣
6model/qty_seq/while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOpAmodel_qty_seq_while_lstm_cell_31_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0ф
'model/qty_seq/while/lstm_cell_31/MatMulMatMul>model/qty_seq/while/TensorArrayV2Read/TensorListGetItem:item:0>model/qty_seq/while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╜
8model/qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOpCmodel_qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0╦
)model/qty_seq/while/lstm_cell_31/MatMul_1MatMul!model_qty_seq_while_placeholder_2@model/qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╚
$model/qty_seq/while/lstm_cell_31/addAddV21model/qty_seq/while/lstm_cell_31/MatMul:product:03model/qty_seq/while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         А╖
7model/qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOpBmodel_qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╤
(model/qty_seq/while/lstm_cell_31/BiasAddBiasAdd(model/qty_seq/while/lstm_cell_31/add:z:0?model/qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аr
0model/qty_seq/while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Щ
&model/qty_seq/while/lstm_cell_31/splitSplit9model/qty_seq/while/lstm_cell_31/split/split_dim:output:01model/qty_seq/while/lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitЦ
(model/qty_seq/while/lstm_cell_31/SigmoidSigmoid/model/qty_seq/while/lstm_cell_31/split:output:0*
T0*'
_output_shapes
:          Ш
*model/qty_seq/while/lstm_cell_31/Sigmoid_1Sigmoid/model/qty_seq/while/lstm_cell_31/split:output:1*
T0*'
_output_shapes
:          ░
$model/qty_seq/while/lstm_cell_31/mulMul.model/qty_seq/while/lstm_cell_31/Sigmoid_1:y:0!model_qty_seq_while_placeholder_3*
T0*'
_output_shapes
:          Р
%model/qty_seq/while/lstm_cell_31/ReluRelu/model/qty_seq/while/lstm_cell_31/split:output:2*
T0*'
_output_shapes
:          ┬
&model/qty_seq/while/lstm_cell_31/mul_1Mul,model/qty_seq/while/lstm_cell_31/Sigmoid:y:03model/qty_seq/while/lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          ╖
&model/qty_seq/while/lstm_cell_31/add_1AddV2(model/qty_seq/while/lstm_cell_31/mul:z:0*model/qty_seq/while/lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          Ш
*model/qty_seq/while/lstm_cell_31/Sigmoid_2Sigmoid/model/qty_seq/while/lstm_cell_31/split:output:3*
T0*'
_output_shapes
:          Н
'model/qty_seq/while/lstm_cell_31/Relu_1Relu*model/qty_seq/while/lstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          ╞
&model/qty_seq/while/lstm_cell_31/mul_2Mul.model/qty_seq/while/lstm_cell_31/Sigmoid_2:y:05model/qty_seq/while/lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          ¤
8model/qty_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!model_qty_seq_while_placeholder_1model_qty_seq_while_placeholder*model/qty_seq/while/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥[
model/qty_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
model/qty_seq/while/addAddV2model_qty_seq_while_placeholder"model/qty_seq/while/add/y:output:0*
T0*
_output_shapes
: ]
model/qty_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Я
model/qty_seq/while/add_1AddV24model_qty_seq_while_model_qty_seq_while_loop_counter$model/qty_seq/while/add_1/y:output:0*
T0*
_output_shapes
: Г
model/qty_seq/while/IdentityIdentitymodel/qty_seq/while/add_1:z:0^model/qty_seq/while/NoOp*
T0*
_output_shapes
: в
model/qty_seq/while/Identity_1Identity:model_qty_seq_while_model_qty_seq_while_maximum_iterations^model/qty_seq/while/NoOp*
T0*
_output_shapes
: Г
model/qty_seq/while/Identity_2Identitymodel/qty_seq/while/add:z:0^model/qty_seq/while/NoOp*
T0*
_output_shapes
: ├
model/qty_seq/while/Identity_3IdentityHmodel/qty_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/qty_seq/while/NoOp*
T0*
_output_shapes
: :щш╥г
model/qty_seq/while/Identity_4Identity*model/qty_seq/while/lstm_cell_31/mul_2:z:0^model/qty_seq/while/NoOp*
T0*'
_output_shapes
:          г
model/qty_seq/while/Identity_5Identity*model/qty_seq/while/lstm_cell_31/add_1:z:0^model/qty_seq/while/NoOp*
T0*'
_output_shapes
:          И
model/qty_seq/while/NoOpNoOp8^model/qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOp7^model/qty_seq/while/lstm_cell_31/MatMul/ReadVariableOp9^model/qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
model_qty_seq_while_identity%model/qty_seq/while/Identity:output:0"I
model_qty_seq_while_identity_1'model/qty_seq/while/Identity_1:output:0"I
model_qty_seq_while_identity_2'model/qty_seq/while/Identity_2:output:0"I
model_qty_seq_while_identity_3'model/qty_seq/while/Identity_3:output:0"I
model_qty_seq_while_identity_4'model/qty_seq/while/Identity_4:output:0"I
model_qty_seq_while_identity_5'model/qty_seq/while/Identity_5:output:0"Ж
@model_qty_seq_while_lstm_cell_31_biasadd_readvariableop_resourceBmodel_qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource_0"И
Amodel_qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resourceCmodel_qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource_0"Д
?model_qty_seq_while_lstm_cell_31_matmul_readvariableop_resourceAmodel_qty_seq_while_lstm_cell_31_matmul_readvariableop_resource_0"h
1model_qty_seq_while_model_qty_seq_strided_slice_13model_qty_seq_while_model_qty_seq_strided_slice_1_0"р
mmodel_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_qty_seq_tensorarrayunstack_tensorlistfromtensoromodel_qty_seq_while_tensorarrayv2read_tensorlistgetitem_model_qty_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2r
7model/qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOp7model/qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOp2p
6model/qty_seq/while/lstm_cell_31/MatMul/ReadVariableOp6model/qty_seq/while/lstm_cell_31/MatMul/ReadVariableOp2t
8model/qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOp8model/qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
А

e
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_126676

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ш
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:          Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
иJ
Ы
C__inference_qty_seq_layer_call_and_return_conditional_losses_127000

inputs>
+lstm_cell_31_matmul_readvariableop_resource:	А@
-lstm_cell_31_matmul_1_readvariableop_resource:	 А;
,lstm_cell_31_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_31/BiasAdd/ReadVariableOpв"lstm_cell_31/MatMul/ReadVariableOpв$lstm_cell_31/MatMul_1/ReadVariableOpвwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskП
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АМ
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АН
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А^
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▌
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitn
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*'
_output_shapes
:          p
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*'
_output_shapes
:          w
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          h
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ж
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          {
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          p
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*'
_output_shapes
:          e
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          К
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_126916*
condR
while_cond_126915*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          └
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Я8
╨
while_body_126315
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_31_matmul_readvariableop_resource_0:	АH
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:	 АC
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_31_matmul_readvariableop_resource:	АF
3while_lstm_cell_31_matmul_1_readvariableop_resource:	 АA
2while_lstm_cell_31_biasadd_readvariableop_resource:	АИв)while/lstm_cell_31/BiasAdd/ReadVariableOpв(while/lstm_cell_31/MatMul/ReadVariableOpв*while/lstm_cell_31/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Э
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0║
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0б
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АЫ
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0з
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitz
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*'
_output_shapes
:          |
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*'
_output_shapes
:          Ж
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          t
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ш
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          Н
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          |
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*'
_output_shapes
:          q
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          Ь
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥y
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          y
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          ╨

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
┘
d
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_126562

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:          [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╡
├
while_cond_128361
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_128361___redundant_placeholder04
0while_while_cond_128361___redundant_placeholder14
0while_while_cond_128361___redundant_placeholder24
0while_while_cond_128361___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╡
├
while_cond_128075
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_128075___redundant_placeholder04
0while_while_cond_128075___redundant_placeholder14
0while_while_cond_128075___redundant_placeholder24
0while_while_cond_128075___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
▌
Ж
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_129214

inputs
states_0
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
иJ
Ы
C__inference_qty_seq_layer_call_and_return_conditional_losses_129062

inputs>
+lstm_cell_31_matmul_readvariableop_resource:	А@
-lstm_cell_31_matmul_1_readvariableop_resource:	 А;
,lstm_cell_31_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_31/BiasAdd/ReadVariableOpв"lstm_cell_31/MatMul/ReadVariableOpв$lstm_cell_31/MatMul_1/ReadVariableOpвwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskП
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АМ
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АН
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А^
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▌
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitn
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*'
_output_shapes
:          p
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*'
_output_shapes
:          w
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          h
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ж
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          {
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          p
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*'
_output_shapes
:          e
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          К
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_128978*
condR
while_cond_128977*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          └
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
иJ
Ы
C__inference_qty_seq_layer_call_and_return_conditional_losses_126399

inputs>
+lstm_cell_31_matmul_readvariableop_resource:	А@
-lstm_cell_31_matmul_1_readvariableop_resource:	 А;
,lstm_cell_31_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_31/BiasAdd/ReadVariableOpв"lstm_cell_31/MatMul/ReadVariableOpв$lstm_cell_31/MatMul_1/ReadVariableOpвwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskП
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АМ
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АН
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А^
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▌
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitn
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*'
_output_shapes
:          p
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*'
_output_shapes
:          w
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          h
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ж
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          {
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          p
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*'
_output_shapes
:          e
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          К
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_126315*
condR
while_cond_126314*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          └
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
щ

╒
&__inference_model_layer_call_fn_126616
	salt_data
quantity_data
unknown:	А
	unknown_0:	 А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	 А
	unknown_4:	А
	unknown_5:@
	unknown_6:
identityИвStatefulPartitionedCall╖
StatefulPartitionedCallStatefulPartitionedCall	salt_dataquantity_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_126597o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:         
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:         
'
_user_specified_namequantity_data
ў
╡
(__inference_qty_seq_layer_call_fn_128490

inputs
unknown:	А
	unknown_0:	 А
	unknown_1:	А
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_127000o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
▌J
Э
C__inference_qty_seq_layer_call_and_return_conditional_losses_128776
inputs_0>
+lstm_cell_31_matmul_readvariableop_resource:	А@
-lstm_cell_31_matmul_1_readvariableop_resource:	 А;
,lstm_cell_31_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_31/BiasAdd/ReadVariableOpв"lstm_cell_31/MatMul/ReadVariableOpв$lstm_cell_31/MatMul_1/ReadVariableOpвwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskП
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АМ
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АН
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А^
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▌
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitn
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*'
_output_shapes
:          p
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*'
_output_shapes
:          w
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          h
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ж
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          {
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          p
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*'
_output_shapes
:          e
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          К
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_128692*
condR
while_cond_128691*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          └
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
╡
├
while_cond_126314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_126314___redundant_placeholder04
0while_while_cond_126314___redundant_placeholder14
0while_while_cond_126314___redundant_placeholder24
0while_while_cond_126314___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
Я
F
*__inference_qty_seq_2_layer_call_fn_129094

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_126569`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
▌
Ж
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_129312

inputs
states_0
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
П▓
У
A__inference_model_layer_call_and_return_conditional_losses_127499
inputs_0
inputs_1F
3qty_seq_lstm_cell_31_matmul_readvariableop_resource:	АH
5qty_seq_lstm_cell_31_matmul_1_readvariableop_resource:	 АC
4qty_seq_lstm_cell_31_biasadd_readvariableop_resource:	АG
4salt_seq_lstm_cell_30_matmul_readvariableop_resource:	АI
6salt_seq_lstm_cell_30_matmul_1_readvariableop_resource:	 АD
5salt_seq_lstm_cell_30_biasadd_readvariableop_resource:	А:
(salt_pred_matmul_readvariableop_resource:@7
)salt_pred_biasadd_readvariableop_resource:
identityИв+qty_seq/lstm_cell_31/BiasAdd/ReadVariableOpв*qty_seq/lstm_cell_31/MatMul/ReadVariableOpв,qty_seq/lstm_cell_31/MatMul_1/ReadVariableOpвqty_seq/whileв salt_pred/BiasAdd/ReadVariableOpвsalt_pred/MatMul/ReadVariableOpв,salt_seq/lstm_cell_30/BiasAdd/ReadVariableOpв+salt_seq/lstm_cell_30/MatMul/ReadVariableOpв-salt_seq/lstm_cell_30/MatMul_1/ReadVariableOpвsalt_seq/whileE
qty_seq/ShapeShapeinputs_1*
T0*
_output_shapes
:e
qty_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
qty_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
qty_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
qty_seq/strided_sliceStridedSliceqty_seq/Shape:output:0$qty_seq/strided_slice/stack:output:0&qty_seq/strided_slice/stack_1:output:0&qty_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
qty_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Л
qty_seq/zeros/packedPackqty_seq/strided_slice:output:0qty_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
qty_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Д
qty_seq/zerosFillqty_seq/zeros/packed:output:0qty_seq/zeros/Const:output:0*
T0*'
_output_shapes
:          Z
qty_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : П
qty_seq/zeros_1/packedPackqty_seq/strided_slice:output:0!qty_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
qty_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    К
qty_seq/zeros_1Fillqty_seq/zeros_1/packed:output:0qty_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:          k
qty_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
qty_seq/transpose	Transposeinputs_1qty_seq/transpose/perm:output:0*
T0*+
_output_shapes
:         T
qty_seq/Shape_1Shapeqty_seq/transpose:y:0*
T0*
_output_shapes
:g
qty_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
qty_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
qty_seq/strided_slice_1StridedSliceqty_seq/Shape_1:output:0&qty_seq/strided_slice_1/stack:output:0(qty_seq/strided_slice_1/stack_1:output:0(qty_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#qty_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
qty_seq/TensorArrayV2TensorListReserve,qty_seq/TensorArrayV2/element_shape:output:0 qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥О
=qty_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       °
/qty_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorqty_seq/transpose:y:0Fqty_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥g
qty_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
qty_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:С
qty_seq/strided_slice_2StridedSliceqty_seq/transpose:y:0&qty_seq/strided_slice_2/stack:output:0(qty_seq/strided_slice_2/stack_1:output:0(qty_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЯ
*qty_seq/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3qty_seq_lstm_cell_31_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0о
qty_seq/lstm_cell_31/MatMulMatMul qty_seq/strided_slice_2:output:02qty_seq/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аг
,qty_seq/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5qty_seq_lstm_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0и
qty_seq/lstm_cell_31/MatMul_1MatMulqty_seq/zeros:output:04qty_seq/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ад
qty_seq/lstm_cell_31/addAddV2%qty_seq/lstm_cell_31/MatMul:product:0'qty_seq/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АЭ
+qty_seq/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4qty_seq_lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0н
qty_seq/lstm_cell_31/BiasAddBiasAddqty_seq/lstm_cell_31/add:z:03qty_seq/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аf
$qty_seq/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
qty_seq/lstm_cell_31/splitSplit-qty_seq/lstm_cell_31/split/split_dim:output:0%qty_seq/lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split~
qty_seq/lstm_cell_31/SigmoidSigmoid#qty_seq/lstm_cell_31/split:output:0*
T0*'
_output_shapes
:          А
qty_seq/lstm_cell_31/Sigmoid_1Sigmoid#qty_seq/lstm_cell_31/split:output:1*
T0*'
_output_shapes
:          П
qty_seq/lstm_cell_31/mulMul"qty_seq/lstm_cell_31/Sigmoid_1:y:0qty_seq/zeros_1:output:0*
T0*'
_output_shapes
:          x
qty_seq/lstm_cell_31/ReluRelu#qty_seq/lstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ю
qty_seq/lstm_cell_31/mul_1Mul qty_seq/lstm_cell_31/Sigmoid:y:0'qty_seq/lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          У
qty_seq/lstm_cell_31/add_1AddV2qty_seq/lstm_cell_31/mul:z:0qty_seq/lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          А
qty_seq/lstm_cell_31/Sigmoid_2Sigmoid#qty_seq/lstm_cell_31/split:output:3*
T0*'
_output_shapes
:          u
qty_seq/lstm_cell_31/Relu_1Reluqty_seq/lstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          в
qty_seq/lstm_cell_31/mul_2Mul"qty_seq/lstm_cell_31/Sigmoid_2:y:0)qty_seq/lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          v
%qty_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╨
qty_seq/TensorArrayV2_1TensorListReserve.qty_seq/TensorArrayV2_1/element_shape:output:0 qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥N
qty_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 qty_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
qty_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Є
qty_seq/whileWhile#qty_seq/while/loop_counter:output:0)qty_seq/while/maximum_iterations:output:0qty_seq/time:output:0 qty_seq/TensorArrayV2_1:handle:0qty_seq/zeros:output:0qty_seq/zeros_1:output:0 qty_seq/strided_slice_1:output:0?qty_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:03qty_seq_lstm_cell_31_matmul_readvariableop_resource5qty_seq_lstm_cell_31_matmul_1_readvariableop_resource4qty_seq_lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
qty_seq_while_body_127266*%
condR
qty_seq_while_cond_127265*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Й
8qty_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┌
*qty_seq/TensorArrayV2Stack/TensorListStackTensorListStackqty_seq/while:output:3Aqty_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0p
qty_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
qty_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
qty_seq/strided_slice_3StridedSlice3qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0&qty_seq/strided_slice_3/stack:output:0(qty_seq/strided_slice_3/stack_1:output:0(qty_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskm
qty_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          о
qty_seq/transpose_1	Transpose3qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0!qty_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:          c
qty_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    F
salt_seq/ShapeShapeinputs_0*
T0*
_output_shapes
:f
salt_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
salt_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
salt_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
salt_seq/strided_sliceStridedSlicesalt_seq/Shape:output:0%salt_seq/strided_slice/stack:output:0'salt_seq/strided_slice/stack_1:output:0'salt_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
salt_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : О
salt_seq/zeros/packedPacksalt_seq/strided_slice:output:0 salt_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
salt_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    З
salt_seq/zerosFillsalt_seq/zeros/packed:output:0salt_seq/zeros/Const:output:0*
T0*'
_output_shapes
:          [
salt_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Т
salt_seq/zeros_1/packedPacksalt_seq/strided_slice:output:0"salt_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
salt_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Н
salt_seq/zeros_1Fill salt_seq/zeros_1/packed:output:0salt_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:          l
salt_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Б
salt_seq/transpose	Transposeinputs_0 salt_seq/transpose/perm:output:0*
T0*+
_output_shapes
:         V
salt_seq/Shape_1Shapesalt_seq/transpose:y:0*
T0*
_output_shapes
:h
salt_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 salt_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
salt_seq/strided_slice_1StridedSlicesalt_seq/Shape_1:output:0'salt_seq/strided_slice_1/stack:output:0)salt_seq/strided_slice_1/stack_1:output:0)salt_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$salt_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╧
salt_seq/TensorArrayV2TensorListReserve-salt_seq/TensorArrayV2/element_shape:output:0!salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥П
>salt_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       √
0salt_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsalt_seq/transpose:y:0Gsalt_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥h
salt_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 salt_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ц
salt_seq/strided_slice_2StridedSlicesalt_seq/transpose:y:0'salt_seq/strided_slice_2/stack:output:0)salt_seq/strided_slice_2/stack_1:output:0)salt_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskб
+salt_seq/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp4salt_seq_lstm_cell_30_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0▒
salt_seq/lstm_cell_30/MatMulMatMul!salt_seq/strided_slice_2:output:03salt_seq/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
-salt_seq/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp6salt_seq_lstm_cell_30_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0л
salt_seq/lstm_cell_30/MatMul_1MatMulsalt_seq/zeros:output:05salt_seq/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
salt_seq/lstm_cell_30/addAddV2&salt_seq/lstm_cell_30/MatMul:product:0(salt_seq/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АЯ
,salt_seq/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp5salt_seq_lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
salt_seq/lstm_cell_30/BiasAddBiasAddsalt_seq/lstm_cell_30/add:z:04salt_seq/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аg
%salt_seq/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :°
salt_seq/lstm_cell_30/splitSplit.salt_seq/lstm_cell_30/split/split_dim:output:0&salt_seq/lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitА
salt_seq/lstm_cell_30/SigmoidSigmoid$salt_seq/lstm_cell_30/split:output:0*
T0*'
_output_shapes
:          В
salt_seq/lstm_cell_30/Sigmoid_1Sigmoid$salt_seq/lstm_cell_30/split:output:1*
T0*'
_output_shapes
:          Т
salt_seq/lstm_cell_30/mulMul#salt_seq/lstm_cell_30/Sigmoid_1:y:0salt_seq/zeros_1:output:0*
T0*'
_output_shapes
:          z
salt_seq/lstm_cell_30/ReluRelu$salt_seq/lstm_cell_30/split:output:2*
T0*'
_output_shapes
:          б
salt_seq/lstm_cell_30/mul_1Mul!salt_seq/lstm_cell_30/Sigmoid:y:0(salt_seq/lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          Ц
salt_seq/lstm_cell_30/add_1AddV2salt_seq/lstm_cell_30/mul:z:0salt_seq/lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          В
salt_seq/lstm_cell_30/Sigmoid_2Sigmoid$salt_seq/lstm_cell_30/split:output:3*
T0*'
_output_shapes
:          w
salt_seq/lstm_cell_30/Relu_1Relusalt_seq/lstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          е
salt_seq/lstm_cell_30/mul_2Mul#salt_seq/lstm_cell_30/Sigmoid_2:y:0*salt_seq/lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          w
&salt_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╙
salt_seq/TensorArrayV2_1TensorListReserve/salt_seq/TensorArrayV2_1/element_shape:output:0!salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥O
salt_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!salt_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         ]
salt_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : А
salt_seq/whileWhile$salt_seq/while/loop_counter:output:0*salt_seq/while/maximum_iterations:output:0salt_seq/time:output:0!salt_seq/TensorArrayV2_1:handle:0salt_seq/zeros:output:0salt_seq/zeros_1:output:0!salt_seq/strided_slice_1:output:0@salt_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:04salt_seq_lstm_cell_30_matmul_readvariableop_resource6salt_seq_lstm_cell_30_matmul_1_readvariableop_resource5salt_seq_lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
salt_seq_while_body_127405*&
condR
salt_seq_while_cond_127404*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations К
9salt_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ▌
+salt_seq/TensorArrayV2Stack/TensorListStackTensorListStacksalt_seq/while:output:3Bsalt_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0q
salt_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         j
 salt_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
salt_seq/strided_slice_3StridedSlice4salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0'salt_seq/strided_slice_3/stack:output:0)salt_seq/strided_slice_3/stack_1:output:0)salt_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskn
salt_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ▒
salt_seq/transpose_1	Transpose4salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0"salt_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:          d
salt_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    t
salt_seq_2/IdentityIdentity!salt_seq/strided_slice_3:output:0*
T0*'
_output_shapes
:          r
qty_seq_2/IdentityIdentity qty_seq/strided_slice_3:output:0*
T0*'
_output_shapes
:          U
pattern/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :о
pattern/concatConcatV2salt_seq_2/Identity:output:0qty_seq_2/Identity:output:0pattern/concat/axis:output:0*
N*
T0*'
_output_shapes
:         @И
salt_pred/MatMul/ReadVariableOpReadVariableOp(salt_pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0О
salt_pred/MatMulMatMulpattern/concat:output:0'salt_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 salt_pred/BiasAdd/ReadVariableOpReadVariableOp)salt_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
salt_pred/BiasAddBiasAddsalt_pred/MatMul:product:0(salt_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         i
IdentityIdentitysalt_pred/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ├
NoOpNoOp,^qty_seq/lstm_cell_31/BiasAdd/ReadVariableOp+^qty_seq/lstm_cell_31/MatMul/ReadVariableOp-^qty_seq/lstm_cell_31/MatMul_1/ReadVariableOp^qty_seq/while!^salt_pred/BiasAdd/ReadVariableOp ^salt_pred/MatMul/ReadVariableOp-^salt_seq/lstm_cell_30/BiasAdd/ReadVariableOp,^salt_seq/lstm_cell_30/MatMul/ReadVariableOp.^salt_seq/lstm_cell_30/MatMul_1/ReadVariableOp^salt_seq/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         :         : : : : : : : : 2Z
+qty_seq/lstm_cell_31/BiasAdd/ReadVariableOp+qty_seq/lstm_cell_31/BiasAdd/ReadVariableOp2X
*qty_seq/lstm_cell_31/MatMul/ReadVariableOp*qty_seq/lstm_cell_31/MatMul/ReadVariableOp2\
,qty_seq/lstm_cell_31/MatMul_1/ReadVariableOp,qty_seq/lstm_cell_31/MatMul_1/ReadVariableOp2
qty_seq/whileqty_seq/while2D
 salt_pred/BiasAdd/ReadVariableOp salt_pred/BiasAdd/ReadVariableOp2B
salt_pred/MatMul/ReadVariableOpsalt_pred/MatMul/ReadVariableOp2\
,salt_seq/lstm_cell_30/BiasAdd/ReadVariableOp,salt_seq/lstm_cell_30/BiasAdd/ReadVariableOp2Z
+salt_seq/lstm_cell_30/MatMul/ReadVariableOp+salt_seq/lstm_cell_30/MatMul/ReadVariableOp2^
-salt_seq/lstm_cell_30/MatMul_1/ReadVariableOp-salt_seq/lstm_cell_30/MatMul_1/ReadVariableOp2 
salt_seq/whilesalt_seq/while:U Q
+
_output_shapes
:         
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/1
Я8
╨
while_body_128076
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_30_matmul_readvariableop_resource_0:	АH
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:	 АC
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_30_matmul_readvariableop_resource:	АF
3while_lstm_cell_30_matmul_1_readvariableop_resource:	 АA
2while_lstm_cell_30_biasadd_readvariableop_resource:	АИв)while/lstm_cell_30/BiasAdd/ReadVariableOpв(while/lstm_cell_30/MatMul/ReadVariableOpв*while/lstm_cell_30/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Э
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0║
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0б
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АЫ
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0з
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitz
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*'
_output_shapes
:          |
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*'
_output_shapes
:          Ж
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          t
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*'
_output_shapes
:          Ш
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          Н
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          |
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*'
_output_shapes
:          q
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          Ь
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥y
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          y
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          ╨

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
Я8
╨
while_body_126916
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_31_matmul_readvariableop_resource_0:	АH
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:	 АC
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_31_matmul_readvariableop_resource:	АF
3while_lstm_cell_31_matmul_1_readvariableop_resource:	 АA
2while_lstm_cell_31_biasadd_readvariableop_resource:	АИв)while/lstm_cell_31/BiasAdd/ReadVariableOpв(while/lstm_cell_31/MatMul/ReadVariableOpв*while/lstm_cell_31/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Э
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0║
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0б
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АЫ
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0з
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitz
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*'
_output_shapes
:          |
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*'
_output_shapes
:          Ж
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          t
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ш
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          Н
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          |
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*'
_output_shapes
:          q
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          Ь
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥y
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          y
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          ╨

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
─
Ч
*__inference_salt_pred_layer_call_fn_129138

inputs
unknown:@
	unknown_0:
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_126590o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┤
m
C__inference_pattern_layer_call_and_return_conditional_losses_126578

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         @W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:          :          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs:OK
'
_output_shapes
:          
 
_user_specified_nameinputs
╪
c
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_126569

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:          [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
▐J
Ю
D__inference_salt_seq_layer_call_and_return_conditional_losses_128160
inputs_0>
+lstm_cell_30_matmul_readvariableop_resource:	А@
-lstm_cell_30_matmul_1_readvariableop_resource:	 А;
,lstm_cell_30_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_30/BiasAdd/ReadVariableOpв"lstm_cell_30/MatMul/ReadVariableOpв$lstm_cell_30/MatMul_1/ReadVariableOpвwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskП
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АМ
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АН
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А^
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▌
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitn
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*'
_output_shapes
:          p
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*'
_output_shapes
:          w
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          h
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*'
_output_shapes
:          Ж
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          {
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          p
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*'
_output_shapes
:          e
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          К
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_128076*
condR
while_cond_128075*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          └
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
йJ
Ь
D__inference_salt_seq_layer_call_and_return_conditional_losses_128446

inputs>
+lstm_cell_30_matmul_readvariableop_resource:	А@
-lstm_cell_30_matmul_1_readvariableop_resource:	 А;
,lstm_cell_30_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_30/BiasAdd/ReadVariableOpв"lstm_cell_30/MatMul/ReadVariableOpв$lstm_cell_30/MatMul_1/ReadVariableOpвwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskП
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АМ
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АН
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А^
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▌
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitn
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*'
_output_shapes
:          p
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*'
_output_shapes
:          w
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          h
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*'
_output_shapes
:          Ж
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          {
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          p
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*'
_output_shapes
:          e
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          К
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_128362*
condR
while_cond_128361*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          └
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╡
├
while_cond_128218
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_128218___redundant_placeholder04
0while_while_cond_128218___redundant_placeholder14
0while_while_cond_128218___redundant_placeholder24
0while_while_cond_128218___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
Ж┬
х
!__inference__wrapped_model_125548
	salt_data
quantity_dataL
9model_qty_seq_lstm_cell_31_matmul_readvariableop_resource:	АN
;model_qty_seq_lstm_cell_31_matmul_1_readvariableop_resource:	 АI
:model_qty_seq_lstm_cell_31_biasadd_readvariableop_resource:	АM
:model_salt_seq_lstm_cell_30_matmul_readvariableop_resource:	АO
<model_salt_seq_lstm_cell_30_matmul_1_readvariableop_resource:	 АJ
;model_salt_seq_lstm_cell_30_biasadd_readvariableop_resource:	А@
.model_salt_pred_matmul_readvariableop_resource:@=
/model_salt_pred_biasadd_readvariableop_resource:
identityИв1model/qty_seq/lstm_cell_31/BiasAdd/ReadVariableOpв0model/qty_seq/lstm_cell_31/MatMul/ReadVariableOpв2model/qty_seq/lstm_cell_31/MatMul_1/ReadVariableOpвmodel/qty_seq/whileв&model/salt_pred/BiasAdd/ReadVariableOpв%model/salt_pred/MatMul/ReadVariableOpв2model/salt_seq/lstm_cell_30/BiasAdd/ReadVariableOpв1model/salt_seq/lstm_cell_30/MatMul/ReadVariableOpв3model/salt_seq/lstm_cell_30/MatMul_1/ReadVariableOpвmodel/salt_seq/whileP
model/qty_seq/ShapeShapequantity_data*
T0*
_output_shapes
:k
!model/qty_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/qty_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/qty_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
model/qty_seq/strided_sliceStridedSlicemodel/qty_seq/Shape:output:0*model/qty_seq/strided_slice/stack:output:0,model/qty_seq/strided_slice/stack_1:output:0,model/qty_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
model/qty_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Э
model/qty_seq/zeros/packedPack$model/qty_seq/strided_slice:output:0%model/qty_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
model/qty_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
model/qty_seq/zerosFill#model/qty_seq/zeros/packed:output:0"model/qty_seq/zeros/Const:output:0*
T0*'
_output_shapes
:          `
model/qty_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : б
model/qty_seq/zeros_1/packedPack$model/qty_seq/strided_slice:output:0'model/qty_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:`
model/qty_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ь
model/qty_seq/zeros_1Fill%model/qty_seq/zeros_1/packed:output:0$model/qty_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:          q
model/qty_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Р
model/qty_seq/transpose	Transposequantity_data%model/qty_seq/transpose/perm:output:0*
T0*+
_output_shapes
:         `
model/qty_seq/Shape_1Shapemodel/qty_seq/transpose:y:0*
T0*
_output_shapes
:m
#model/qty_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/qty_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/qty_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
model/qty_seq/strided_slice_1StridedSlicemodel/qty_seq/Shape_1:output:0,model/qty_seq/strided_slice_1/stack:output:0.model/qty_seq/strided_slice_1/stack_1:output:0.model/qty_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)model/qty_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▐
model/qty_seq/TensorArrayV2TensorListReserve2model/qty_seq/TensorArrayV2/element_shape:output:0&model/qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ф
Cmodel/qty_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       К
5model/qty_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/qty_seq/transpose:y:0Lmodel/qty_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥m
#model/qty_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/qty_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/qty_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
model/qty_seq/strided_slice_2StridedSlicemodel/qty_seq/transpose:y:0,model/qty_seq/strided_slice_2/stack:output:0.model/qty_seq/strided_slice_2/stack_1:output:0.model/qty_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskл
0model/qty_seq/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp9model_qty_seq_lstm_cell_31_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0└
!model/qty_seq/lstm_cell_31/MatMulMatMul&model/qty_seq/strided_slice_2:output:08model/qty_seq/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ап
2model/qty_seq/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp;model_qty_seq_lstm_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0║
#model/qty_seq/lstm_cell_31/MatMul_1MatMulmodel/qty_seq/zeros:output:0:model/qty_seq/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╢
model/qty_seq/lstm_cell_31/addAddV2+model/qty_seq/lstm_cell_31/MatMul:product:0-model/qty_seq/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         Ай
1model/qty_seq/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp:model_qty_seq_lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0┐
"model/qty_seq/lstm_cell_31/BiasAddBiasAdd"model/qty_seq/lstm_cell_31/add:z:09model/qty_seq/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аl
*model/qty_seq/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :З
 model/qty_seq/lstm_cell_31/splitSplit3model/qty_seq/lstm_cell_31/split/split_dim:output:0+model/qty_seq/lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitК
"model/qty_seq/lstm_cell_31/SigmoidSigmoid)model/qty_seq/lstm_cell_31/split:output:0*
T0*'
_output_shapes
:          М
$model/qty_seq/lstm_cell_31/Sigmoid_1Sigmoid)model/qty_seq/lstm_cell_31/split:output:1*
T0*'
_output_shapes
:          б
model/qty_seq/lstm_cell_31/mulMul(model/qty_seq/lstm_cell_31/Sigmoid_1:y:0model/qty_seq/zeros_1:output:0*
T0*'
_output_shapes
:          Д
model/qty_seq/lstm_cell_31/ReluRelu)model/qty_seq/lstm_cell_31/split:output:2*
T0*'
_output_shapes
:          ░
 model/qty_seq/lstm_cell_31/mul_1Mul&model/qty_seq/lstm_cell_31/Sigmoid:y:0-model/qty_seq/lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          е
 model/qty_seq/lstm_cell_31/add_1AddV2"model/qty_seq/lstm_cell_31/mul:z:0$model/qty_seq/lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          М
$model/qty_seq/lstm_cell_31/Sigmoid_2Sigmoid)model/qty_seq/lstm_cell_31/split:output:3*
T0*'
_output_shapes
:          Б
!model/qty_seq/lstm_cell_31/Relu_1Relu$model/qty_seq/lstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          ┤
 model/qty_seq/lstm_cell_31/mul_2Mul(model/qty_seq/lstm_cell_31/Sigmoid_2:y:0/model/qty_seq/lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          |
+model/qty_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        т
model/qty_seq/TensorArrayV2_1TensorListReserve4model/qty_seq/TensorArrayV2_1/element_shape:output:0&model/qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥T
model/qty_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&model/qty_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         b
 model/qty_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╞
model/qty_seq/whileWhile)model/qty_seq/while/loop_counter:output:0/model/qty_seq/while/maximum_iterations:output:0model/qty_seq/time:output:0&model/qty_seq/TensorArrayV2_1:handle:0model/qty_seq/zeros:output:0model/qty_seq/zeros_1:output:0&model/qty_seq/strided_slice_1:output:0Emodel/qty_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:09model_qty_seq_lstm_cell_31_matmul_readvariableop_resource;model_qty_seq_lstm_cell_31_matmul_1_readvariableop_resource:model_qty_seq_lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *+
body#R!
model_qty_seq_while_body_125315*+
cond#R!
model_qty_seq_while_cond_125314*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations П
>model/qty_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ь
0model/qty_seq/TensorArrayV2Stack/TensorListStackTensorListStackmodel/qty_seq/while:output:3Gmodel/qty_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0v
#model/qty_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         o
%model/qty_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%model/qty_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:═
model/qty_seq/strided_slice_3StridedSlice9model/qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0,model/qty_seq/strided_slice_3/stack:output:0.model/qty_seq/strided_slice_3/stack_1:output:0.model/qty_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_masks
model/qty_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          └
model/qty_seq/transpose_1	Transpose9model/qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0'model/qty_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:          i
model/qty_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    M
model/salt_seq/ShapeShape	salt_data*
T0*
_output_shapes
:l
"model/salt_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model/salt_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model/salt_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
model/salt_seq/strided_sliceStridedSlicemodel/salt_seq/Shape:output:0+model/salt_seq/strided_slice/stack:output:0-model/salt_seq/strided_slice/stack_1:output:0-model/salt_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
model/salt_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : а
model/salt_seq/zeros/packedPack%model/salt_seq/strided_slice:output:0&model/salt_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
model/salt_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Щ
model/salt_seq/zerosFill$model/salt_seq/zeros/packed:output:0#model/salt_seq/zeros/Const:output:0*
T0*'
_output_shapes
:          a
model/salt_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : д
model/salt_seq/zeros_1/packedPack%model/salt_seq/strided_slice:output:0(model/salt_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:a
model/salt_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Я
model/salt_seq/zeros_1Fill&model/salt_seq/zeros_1/packed:output:0%model/salt_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:          r
model/salt_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          О
model/salt_seq/transpose	Transpose	salt_data&model/salt_seq/transpose/perm:output:0*
T0*+
_output_shapes
:         b
model/salt_seq/Shape_1Shapemodel/salt_seq/transpose:y:0*
T0*
_output_shapes
:n
$model/salt_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model/salt_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model/salt_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
model/salt_seq/strided_slice_1StridedSlicemodel/salt_seq/Shape_1:output:0-model/salt_seq/strided_slice_1/stack:output:0/model/salt_seq/strided_slice_1/stack_1:output:0/model/salt_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*model/salt_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         с
model/salt_seq/TensorArrayV2TensorListReserve3model/salt_seq/TensorArrayV2/element_shape:output:0'model/salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Х
Dmodel/salt_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Н
6model/salt_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/salt_seq/transpose:y:0Mmodel/salt_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥n
$model/salt_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model/salt_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model/salt_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
model/salt_seq/strided_slice_2StridedSlicemodel/salt_seq/transpose:y:0-model/salt_seq/strided_slice_2/stack:output:0/model/salt_seq/strided_slice_2/stack_1:output:0/model/salt_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskн
1model/salt_seq/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp:model_salt_seq_lstm_cell_30_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0├
"model/salt_seq/lstm_cell_30/MatMulMatMul'model/salt_seq/strided_slice_2:output:09model/salt_seq/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А▒
3model/salt_seq/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp<model_salt_seq_lstm_cell_30_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0╜
$model/salt_seq/lstm_cell_30/MatMul_1MatMulmodel/salt_seq/zeros:output:0;model/salt_seq/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╣
model/salt_seq/lstm_cell_30/addAddV2,model/salt_seq/lstm_cell_30/MatMul:product:0.model/salt_seq/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         Ал
2model/salt_seq/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp;model_salt_seq_lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0┬
#model/salt_seq/lstm_cell_30/BiasAddBiasAdd#model/salt_seq/lstm_cell_30/add:z:0:model/salt_seq/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
+model/salt_seq/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :К
!model/salt_seq/lstm_cell_30/splitSplit4model/salt_seq/lstm_cell_30/split/split_dim:output:0,model/salt_seq/lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitМ
#model/salt_seq/lstm_cell_30/SigmoidSigmoid*model/salt_seq/lstm_cell_30/split:output:0*
T0*'
_output_shapes
:          О
%model/salt_seq/lstm_cell_30/Sigmoid_1Sigmoid*model/salt_seq/lstm_cell_30/split:output:1*
T0*'
_output_shapes
:          д
model/salt_seq/lstm_cell_30/mulMul)model/salt_seq/lstm_cell_30/Sigmoid_1:y:0model/salt_seq/zeros_1:output:0*
T0*'
_output_shapes
:          Ж
 model/salt_seq/lstm_cell_30/ReluRelu*model/salt_seq/lstm_cell_30/split:output:2*
T0*'
_output_shapes
:          │
!model/salt_seq/lstm_cell_30/mul_1Mul'model/salt_seq/lstm_cell_30/Sigmoid:y:0.model/salt_seq/lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          и
!model/salt_seq/lstm_cell_30/add_1AddV2#model/salt_seq/lstm_cell_30/mul:z:0%model/salt_seq/lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          О
%model/salt_seq/lstm_cell_30/Sigmoid_2Sigmoid*model/salt_seq/lstm_cell_30/split:output:3*
T0*'
_output_shapes
:          Г
"model/salt_seq/lstm_cell_30/Relu_1Relu%model/salt_seq/lstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          ╖
!model/salt_seq/lstm_cell_30/mul_2Mul)model/salt_seq/lstm_cell_30/Sigmoid_2:y:00model/salt_seq/lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          }
,model/salt_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        х
model/salt_seq/TensorArrayV2_1TensorListReserve5model/salt_seq/TensorArrayV2_1/element_shape:output:0'model/salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥U
model/salt_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'model/salt_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         c
!model/salt_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╘
model/salt_seq/whileWhile*model/salt_seq/while/loop_counter:output:00model/salt_seq/while/maximum_iterations:output:0model/salt_seq/time:output:0'model/salt_seq/TensorArrayV2_1:handle:0model/salt_seq/zeros:output:0model/salt_seq/zeros_1:output:0'model/salt_seq/strided_slice_1:output:0Fmodel/salt_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:0:model_salt_seq_lstm_cell_30_matmul_readvariableop_resource<model_salt_seq_lstm_cell_30_matmul_1_readvariableop_resource;model_salt_seq_lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *,
body$R"
 model_salt_seq_while_body_125454*,
cond$R"
 model_salt_seq_while_cond_125453*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Р
?model/salt_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        я
1model/salt_seq/TensorArrayV2Stack/TensorListStackTensorListStackmodel/salt_seq/while:output:3Hmodel/salt_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0w
$model/salt_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         p
&model/salt_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&model/salt_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╥
model/salt_seq/strided_slice_3StridedSlice:model/salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0-model/salt_seq/strided_slice_3/stack:output:0/model/salt_seq/strided_slice_3/stack_1:output:0/model/salt_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskt
model/salt_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ├
model/salt_seq/transpose_1	Transpose:model/salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0(model/salt_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:          j
model/salt_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    А
model/salt_seq_2/IdentityIdentity'model/salt_seq/strided_slice_3:output:0*
T0*'
_output_shapes
:          ~
model/qty_seq_2/IdentityIdentity&model/qty_seq/strided_slice_3:output:0*
T0*'
_output_shapes
:          [
model/pattern/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╞
model/pattern/concatConcatV2"model/salt_seq_2/Identity:output:0!model/qty_seq_2/Identity:output:0"model/pattern/concat/axis:output:0*
N*
T0*'
_output_shapes
:         @Ф
%model/salt_pred/MatMul/ReadVariableOpReadVariableOp.model_salt_pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0а
model/salt_pred/MatMulMatMulmodel/pattern/concat:output:0-model/salt_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Т
&model/salt_pred/BiasAdd/ReadVariableOpReadVariableOp/model_salt_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ж
model/salt_pred/BiasAddBiasAdd model/salt_pred/MatMul:product:0.model/salt_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         o
IdentityIdentity model/salt_pred/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:          
NoOpNoOp2^model/qty_seq/lstm_cell_31/BiasAdd/ReadVariableOp1^model/qty_seq/lstm_cell_31/MatMul/ReadVariableOp3^model/qty_seq/lstm_cell_31/MatMul_1/ReadVariableOp^model/qty_seq/while'^model/salt_pred/BiasAdd/ReadVariableOp&^model/salt_pred/MatMul/ReadVariableOp3^model/salt_seq/lstm_cell_30/BiasAdd/ReadVariableOp2^model/salt_seq/lstm_cell_30/MatMul/ReadVariableOp4^model/salt_seq/lstm_cell_30/MatMul_1/ReadVariableOp^model/salt_seq/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         :         : : : : : : : : 2f
1model/qty_seq/lstm_cell_31/BiasAdd/ReadVariableOp1model/qty_seq/lstm_cell_31/BiasAdd/ReadVariableOp2d
0model/qty_seq/lstm_cell_31/MatMul/ReadVariableOp0model/qty_seq/lstm_cell_31/MatMul/ReadVariableOp2h
2model/qty_seq/lstm_cell_31/MatMul_1/ReadVariableOp2model/qty_seq/lstm_cell_31/MatMul_1/ReadVariableOp2*
model/qty_seq/whilemodel/qty_seq/while2P
&model/salt_pred/BiasAdd/ReadVariableOp&model/salt_pred/BiasAdd/ReadVariableOp2N
%model/salt_pred/MatMul/ReadVariableOp%model/salt_pred/MatMul/ReadVariableOp2h
2model/salt_seq/lstm_cell_30/BiasAdd/ReadVariableOp2model/salt_seq/lstm_cell_30/BiasAdd/ReadVariableOp2f
1model/salt_seq/lstm_cell_30/MatMul/ReadVariableOp1model/salt_seq/lstm_cell_30/MatMul/ReadVariableOp2j
3model/salt_seq/lstm_cell_30/MatMul_1/ReadVariableOp3model/salt_seq/lstm_cell_30/MatMul_1/ReadVariableOp2,
model/salt_seq/whilemodel/salt_seq/while:V R
+
_output_shapes
:         
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:         
'
_user_specified_namequantity_data
╡
├
while_cond_125819
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_125819___redundant_placeholder04
0while_while_cond_125819___redundant_placeholder14
0while_while_cond_125819___redundant_placeholder24
0while_while_cond_125819___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
є
d
+__inference_salt_seq_2_layer_call_fn_129072

inputs
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_126676o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╡
├
while_cond_125978
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_125978___redundant_placeholder04
0while_while_cond_125978___redundant_placeholder14
0while_while_cond_125978___redundant_placeholder24
0while_while_cond_125978___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╒
Д
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_125761

inputs

states
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates
╡
├
while_cond_126169
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_126169___redundant_placeholder04
0while_while_cond_126169___redundant_placeholder14
0while_while_cond_126169___redundant_placeholder24
0while_while_cond_126169___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
┘
d
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_129077

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:          [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
▌
Ж
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_129344

inputs
states_0
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
╡
├
while_cond_126464
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_126464___redundant_placeholder04
0while_while_cond_126464___redundant_placeholder14
0while_while_cond_126464___redundant_placeholder24
0while_while_cond_126464___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╡
├
while_cond_126915
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_126915___redundant_placeholder04
0while_while_cond_126915___redundant_placeholder14
0while_while_cond_126915___redundant_placeholder24
0while_while_cond_126915___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
иJ
Ы
C__inference_qty_seq_layer_call_and_return_conditional_losses_128919

inputs>
+lstm_cell_31_matmul_readvariableop_resource:	А@
-lstm_cell_31_matmul_1_readvariableop_resource:	 А;
,lstm_cell_31_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_31/BiasAdd/ReadVariableOpв"lstm_cell_31/MatMul/ReadVariableOpв$lstm_cell_31/MatMul_1/ReadVariableOpвwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskП
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АМ
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АН
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А^
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▌
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitn
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*'
_output_shapes
:          p
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*'
_output_shapes
:          w
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          h
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ж
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          {
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          p
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*'
_output_shapes
:          e
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          К
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_128835*
condR
while_cond_128834*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          └
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╪A
╨

qty_seq_while_body_127266,
(qty_seq_while_qty_seq_while_loop_counter2
.qty_seq_while_qty_seq_while_maximum_iterations
qty_seq_while_placeholder
qty_seq_while_placeholder_1
qty_seq_while_placeholder_2
qty_seq_while_placeholder_3+
'qty_seq_while_qty_seq_strided_slice_1_0g
cqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0N
;qty_seq_while_lstm_cell_31_matmul_readvariableop_resource_0:	АP
=qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource_0:	 АK
<qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource_0:	А
qty_seq_while_identity
qty_seq_while_identity_1
qty_seq_while_identity_2
qty_seq_while_identity_3
qty_seq_while_identity_4
qty_seq_while_identity_5)
%qty_seq_while_qty_seq_strided_slice_1e
aqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensorL
9qty_seq_while_lstm_cell_31_matmul_readvariableop_resource:	АN
;qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource:	 АI
:qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource:	АИв1qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOpв0qty_seq/while/lstm_cell_31/MatMul/ReadVariableOpв2qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOpР
?qty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1qty_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemcqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0qty_seq_while_placeholderHqty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0н
0qty_seq/while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp;qty_seq_while_lstm_cell_31_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0╥
!qty_seq/while/lstm_cell_31/MatMulMatMul8qty_seq/while/TensorArrayV2Read/TensorListGetItem:item:08qty_seq/while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А▒
2qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp=qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0╣
#qty_seq/while/lstm_cell_31/MatMul_1MatMulqty_seq_while_placeholder_2:qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╢
qty_seq/while/lstm_cell_31/addAddV2+qty_seq/while/lstm_cell_31/MatMul:product:0-qty_seq/while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         Ал
1qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp<qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0┐
"qty_seq/while/lstm_cell_31/BiasAddBiasAdd"qty_seq/while/lstm_cell_31/add:z:09qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аl
*qty_seq/while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :З
 qty_seq/while/lstm_cell_31/splitSplit3qty_seq/while/lstm_cell_31/split/split_dim:output:0+qty_seq/while/lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitК
"qty_seq/while/lstm_cell_31/SigmoidSigmoid)qty_seq/while/lstm_cell_31/split:output:0*
T0*'
_output_shapes
:          М
$qty_seq/while/lstm_cell_31/Sigmoid_1Sigmoid)qty_seq/while/lstm_cell_31/split:output:1*
T0*'
_output_shapes
:          Ю
qty_seq/while/lstm_cell_31/mulMul(qty_seq/while/lstm_cell_31/Sigmoid_1:y:0qty_seq_while_placeholder_3*
T0*'
_output_shapes
:          Д
qty_seq/while/lstm_cell_31/ReluRelu)qty_seq/while/lstm_cell_31/split:output:2*
T0*'
_output_shapes
:          ░
 qty_seq/while/lstm_cell_31/mul_1Mul&qty_seq/while/lstm_cell_31/Sigmoid:y:0-qty_seq/while/lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          е
 qty_seq/while/lstm_cell_31/add_1AddV2"qty_seq/while/lstm_cell_31/mul:z:0$qty_seq/while/lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          М
$qty_seq/while/lstm_cell_31/Sigmoid_2Sigmoid)qty_seq/while/lstm_cell_31/split:output:3*
T0*'
_output_shapes
:          Б
!qty_seq/while/lstm_cell_31/Relu_1Relu$qty_seq/while/lstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          ┤
 qty_seq/while/lstm_cell_31/mul_2Mul(qty_seq/while/lstm_cell_31/Sigmoid_2:y:0/qty_seq/while/lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          х
2qty_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemqty_seq_while_placeholder_1qty_seq_while_placeholder$qty_seq/while/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥U
qty_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
qty_seq/while/addAddV2qty_seq_while_placeholderqty_seq/while/add/y:output:0*
T0*
_output_shapes
: W
qty_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
qty_seq/while/add_1AddV2(qty_seq_while_qty_seq_while_loop_counterqty_seq/while/add_1/y:output:0*
T0*
_output_shapes
: q
qty_seq/while/IdentityIdentityqty_seq/while/add_1:z:0^qty_seq/while/NoOp*
T0*
_output_shapes
: К
qty_seq/while/Identity_1Identity.qty_seq_while_qty_seq_while_maximum_iterations^qty_seq/while/NoOp*
T0*
_output_shapes
: q
qty_seq/while/Identity_2Identityqty_seq/while/add:z:0^qty_seq/while/NoOp*
T0*
_output_shapes
: ▒
qty_seq/while/Identity_3IdentityBqty_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^qty_seq/while/NoOp*
T0*
_output_shapes
: :щш╥С
qty_seq/while/Identity_4Identity$qty_seq/while/lstm_cell_31/mul_2:z:0^qty_seq/while/NoOp*
T0*'
_output_shapes
:          С
qty_seq/while/Identity_5Identity$qty_seq/while/lstm_cell_31/add_1:z:0^qty_seq/while/NoOp*
T0*'
_output_shapes
:          Ё
qty_seq/while/NoOpNoOp2^qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOp1^qty_seq/while/lstm_cell_31/MatMul/ReadVariableOp3^qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
qty_seq_while_identityqty_seq/while/Identity:output:0"=
qty_seq_while_identity_1!qty_seq/while/Identity_1:output:0"=
qty_seq_while_identity_2!qty_seq/while/Identity_2:output:0"=
qty_seq_while_identity_3!qty_seq/while/Identity_3:output:0"=
qty_seq_while_identity_4!qty_seq/while/Identity_4:output:0"=
qty_seq_while_identity_5!qty_seq/while/Identity_5:output:0"z
:qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource<qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource_0"|
;qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource=qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource_0"x
9qty_seq_while_lstm_cell_31_matmul_readvariableop_resource;qty_seq_while_lstm_cell_31_matmul_readvariableop_resource_0"P
%qty_seq_while_qty_seq_strided_slice_1'qty_seq_while_qty_seq_strided_slice_1_0"╚
aqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensorcqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2f
1qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOp1qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOp2d
0qty_seq/while/lstm_cell_31/MatMul/ReadVariableOp0qty_seq/while/lstm_cell_31/MatMul/ReadVariableOp2h
2qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOp2qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
╝
o
C__inference_pattern_layer_call_and_return_conditional_losses_129129
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         @W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:          :          :Q M
'
_output_shapes
:          
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:          
"
_user_specified_name
inputs/1
Я8
╨
while_body_126465
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_30_matmul_readvariableop_resource_0:	АH
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:	 АC
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_30_matmul_readvariableop_resource:	АF
3while_lstm_cell_30_matmul_1_readvariableop_resource:	 АA
2while_lstm_cell_30_biasadd_readvariableop_resource:	АИв)while/lstm_cell_30/BiasAdd/ReadVariableOpв(while/lstm_cell_30/MatMul/ReadVariableOpв*while/lstm_cell_30/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Э
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0║
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0б
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АЫ
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0з
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitz
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*'
_output_shapes
:          |
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*'
_output_shapes
:          Ж
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          t
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*'
_output_shapes
:          Ш
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          Н
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          |
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*'
_output_shapes
:          q
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          Ь
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥y
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          y
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          ╨

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
∙
╢
)__inference_salt_seq_layer_call_fn_127863

inputs
unknown:	А
	unknown_0:	 А
	unknown_1:	А
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_126549o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Я8
╨
while_body_128362
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_30_matmul_readvariableop_resource_0:	АH
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:	 АC
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_30_matmul_readvariableop_resource:	АF
3while_lstm_cell_30_matmul_1_readvariableop_resource:	 АA
2while_lstm_cell_30_biasadd_readvariableop_resource:	АИв)while/lstm_cell_30/BiasAdd/ReadVariableOpв(while/lstm_cell_30/MatMul/ReadVariableOpв*while/lstm_cell_30/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Э
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0║
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0б
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АЫ
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0з
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitz
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*'
_output_shapes
:          |
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*'
_output_shapes
:          Ж
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          t
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*'
_output_shapes
:          Ш
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          Н
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          |
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*'
_output_shapes
:          q
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          Ь
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥y
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          y
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          ╨

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
∙"
у
while_body_125820
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_30_125844_0:	А.
while_lstm_cell_30_125846_0:	 А*
while_lstm_cell_30_125848_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_30_125844:	А,
while_lstm_cell_30_125846:	 А(
while_lstm_cell_30_125848:	АИв*while/lstm_cell_30/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0│
*while/lstm_cell_30/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_30_125844_0while_lstm_cell_30_125846_0while_lstm_cell_30_125848_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_125761▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_30/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥Р
while/Identity_4Identity3while/lstm_cell_30/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          Р
while/Identity_5Identity3while/lstm_cell_30/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          y

while/NoOpNoOp+^while/lstm_cell_30/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_30_125844while_lstm_cell_30_125844_0"8
while_lstm_cell_30_125846while_lstm_cell_30_125846_0"8
while_lstm_cell_30_125848while_lstm_cell_30_125848_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2X
*while/lstm_cell_30/StatefulPartitionedCall*while/lstm_cell_30/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
╒
Д
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_125965

inputs

states
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates
Щ
╣
A__inference_model_layer_call_and_return_conditional_losses_127156
	salt_data
quantity_data!
qty_seq_127133:	А!
qty_seq_127135:	 А
qty_seq_127137:	А"
salt_seq_127140:	А"
salt_seq_127142:	 А
salt_seq_127144:	А"
salt_pred_127150:@
salt_pred_127152:
identityИвqty_seq/StatefulPartitionedCallв!qty_seq_2/StatefulPartitionedCallв!salt_pred/StatefulPartitionedCallв salt_seq/StatefulPartitionedCallв"salt_seq_2/StatefulPartitionedCallЕ
qty_seq/StatefulPartitionedCallStatefulPartitionedCallquantity_dataqty_seq_127133qty_seq_127135qty_seq_127137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_127000Ж
 salt_seq/StatefulPartitionedCallStatefulPartitionedCall	salt_datasalt_seq_127140salt_seq_127142salt_seq_127144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_126835я
"salt_seq_2/StatefulPartitionedCallStatefulPartitionedCall)salt_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_126676С
!qty_seq_2/StatefulPartitionedCallStatefulPartitionedCall(qty_seq/StatefulPartitionedCall:output:0#^salt_seq_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_126653И
pattern/PartitionedCallPartitionedCall+salt_seq_2/StatefulPartitionedCall:output:0*qty_seq_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_126578О
!salt_pred/StatefulPartitionedCallStatefulPartitionedCall pattern/PartitionedCall:output:0salt_pred_127150salt_pred_127152*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_126590y
IdentityIdentity*salt_pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         °
NoOpNoOp ^qty_seq/StatefulPartitionedCall"^qty_seq_2/StatefulPartitionedCall"^salt_pred/StatefulPartitionedCall!^salt_seq/StatefulPartitionedCall#^salt_seq_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         :         : : : : : : : : 2B
qty_seq/StatefulPartitionedCallqty_seq/StatefulPartitionedCall2F
!qty_seq_2/StatefulPartitionedCall!qty_seq_2/StatefulPartitionedCall2F
!salt_pred/StatefulPartitionedCall!salt_pred/StatefulPartitionedCall2D
 salt_seq/StatefulPartitionedCall salt_seq/StatefulPartitionedCall2H
"salt_seq_2/StatefulPartitionedCall"salt_seq_2/StatefulPartitionedCall:V R
+
_output_shapes
:         
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:         
'
_user_specified_namequantity_data
ы
Ў
-__inference_lstm_cell_30_layer_call_fn_129165

inputs
states_0
states_1
unknown:	А
	unknown_0:	 А
	unknown_1:	А
identity

identity_1

identity_2ИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_125615o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
╡
├
while_cond_127932
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_127932___redundant_placeholder04
0while_while_cond_127932___redundant_placeholder14
0while_while_cond_127932___redundant_placeholder24
0while_while_cond_127932___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
∙
╢
)__inference_salt_seq_layer_call_fn_127874

inputs
unknown:	А
	unknown_0:	 А
	unknown_1:	А
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_126835o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
▓

ў
salt_seq_while_cond_127404.
*salt_seq_while_salt_seq_while_loop_counter4
0salt_seq_while_salt_seq_while_maximum_iterations
salt_seq_while_placeholder 
salt_seq_while_placeholder_1 
salt_seq_while_placeholder_2 
salt_seq_while_placeholder_30
,salt_seq_while_less_salt_seq_strided_slice_1F
Bsalt_seq_while_salt_seq_while_cond_127404___redundant_placeholder0F
Bsalt_seq_while_salt_seq_while_cond_127404___redundant_placeholder1F
Bsalt_seq_while_salt_seq_while_cond_127404___redundant_placeholder2F
Bsalt_seq_while_salt_seq_while_cond_127404___redundant_placeholder3
salt_seq_while_identity
Ж
salt_seq/while/LessLesssalt_seq_while_placeholder,salt_seq_while_less_salt_seq_strided_slice_1*
T0*
_output_shapes
: ]
salt_seq/while/IdentityIdentitysalt_seq/while/Less:z:0*
T0
*
_output_shapes
: ";
salt_seq_while_identity salt_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╡
├
while_cond_128691
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_128691___redundant_placeholder04
0while_while_cond_128691___redundant_placeholder14
0while_while_cond_128691___redundant_placeholder24
0while_while_cond_128691___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
П8
Ж
D__inference_salt_seq_layer_call_and_return_conditional_losses_125698

inputs&
lstm_cell_30_125616:	А&
lstm_cell_30_125618:	 А"
lstm_cell_30_125620:	А
identityИв$lstm_cell_30/StatefulPartitionedCallвwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskї
$lstm_cell_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_30_125616lstm_cell_30_125618lstm_cell_30_125620*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_125615n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╖
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_30_125616lstm_cell_30_125618lstm_cell_30_125620*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_125629*
condR
while_cond_125628*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          u
NoOpNoOp%^lstm_cell_30/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_30/StatefulPartitionedCall$lstm_cell_30/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Я8
╨
while_body_127933
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_30_matmul_readvariableop_resource_0:	АH
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:	 АC
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_30_matmul_readvariableop_resource:	АF
3while_lstm_cell_30_matmul_1_readvariableop_resource:	 АA
2while_lstm_cell_30_biasadd_readvariableop_resource:	АИв)while/lstm_cell_30/BiasAdd/ReadVariableOpв(while/lstm_cell_30/MatMul/ReadVariableOpв*while/lstm_cell_30/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Э
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0║
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0б
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АЫ
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0з
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitz
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*'
_output_shapes
:          |
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*'
_output_shapes
:          Ж
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          t
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*'
_output_shapes
:          Ш
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          Н
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          |
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*'
_output_shapes
:          q
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          Ь
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥y
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          y
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          ╨

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
С
╕
)__inference_salt_seq_layer_call_fn_127852
inputs_0
unknown:	А
	unknown_0:	 А
	unknown_1:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_125889o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
▌J
Э
C__inference_qty_seq_layer_call_and_return_conditional_losses_128633
inputs_0>
+lstm_cell_31_matmul_readvariableop_resource:	А@
-lstm_cell_31_matmul_1_readvariableop_resource:	 А;
,lstm_cell_31_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_31/BiasAdd/ReadVariableOpв"lstm_cell_31/MatMul/ReadVariableOpв$lstm_cell_31/MatMul_1/ReadVariableOpвwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskП
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АМ
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АН
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А^
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▌
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitn
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*'
_output_shapes
:          p
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*'
_output_shapes
:          w
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          h
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ж
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          {
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          p
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*'
_output_shapes
:          e
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          К
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_128549*
condR
while_cond_128548*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          └
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
юB
Ё

salt_seq_while_body_127405.
*salt_seq_while_salt_seq_while_loop_counter4
0salt_seq_while_salt_seq_while_maximum_iterations
salt_seq_while_placeholder 
salt_seq_while_placeholder_1 
salt_seq_while_placeholder_2 
salt_seq_while_placeholder_3-
)salt_seq_while_salt_seq_strided_slice_1_0i
esalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0O
<salt_seq_while_lstm_cell_30_matmul_readvariableop_resource_0:	АQ
>salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource_0:	 АL
=salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource_0:	А
salt_seq_while_identity
salt_seq_while_identity_1
salt_seq_while_identity_2
salt_seq_while_identity_3
salt_seq_while_identity_4
salt_seq_while_identity_5+
'salt_seq_while_salt_seq_strided_slice_1g
csalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensorM
:salt_seq_while_lstm_cell_30_matmul_readvariableop_resource:	АO
<salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource:	 АJ
;salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource:	АИв2salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOpв1salt_seq/while/lstm_cell_30/MatMul/ReadVariableOpв3salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOpС
@salt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╙
2salt_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemesalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0salt_seq_while_placeholderIsalt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0п
1salt_seq/while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp<salt_seq_while_lstm_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0╒
"salt_seq/while/lstm_cell_30/MatMulMatMul9salt_seq/while/TensorArrayV2Read/TensorListGetItem:item:09salt_seq/while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А│
3salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp>salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0╝
$salt_seq/while/lstm_cell_30/MatMul_1MatMulsalt_seq_while_placeholder_2;salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╣
salt_seq/while/lstm_cell_30/addAddV2,salt_seq/while/lstm_cell_30/MatMul:product:0.salt_seq/while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         Ан
2salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp=salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0┬
#salt_seq/while/lstm_cell_30/BiasAddBiasAdd#salt_seq/while/lstm_cell_30/add:z:0:salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
+salt_seq/while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :К
!salt_seq/while/lstm_cell_30/splitSplit4salt_seq/while/lstm_cell_30/split/split_dim:output:0,salt_seq/while/lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitМ
#salt_seq/while/lstm_cell_30/SigmoidSigmoid*salt_seq/while/lstm_cell_30/split:output:0*
T0*'
_output_shapes
:          О
%salt_seq/while/lstm_cell_30/Sigmoid_1Sigmoid*salt_seq/while/lstm_cell_30/split:output:1*
T0*'
_output_shapes
:          б
salt_seq/while/lstm_cell_30/mulMul)salt_seq/while/lstm_cell_30/Sigmoid_1:y:0salt_seq_while_placeholder_3*
T0*'
_output_shapes
:          Ж
 salt_seq/while/lstm_cell_30/ReluRelu*salt_seq/while/lstm_cell_30/split:output:2*
T0*'
_output_shapes
:          │
!salt_seq/while/lstm_cell_30/mul_1Mul'salt_seq/while/lstm_cell_30/Sigmoid:y:0.salt_seq/while/lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          и
!salt_seq/while/lstm_cell_30/add_1AddV2#salt_seq/while/lstm_cell_30/mul:z:0%salt_seq/while/lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          О
%salt_seq/while/lstm_cell_30/Sigmoid_2Sigmoid*salt_seq/while/lstm_cell_30/split:output:3*
T0*'
_output_shapes
:          Г
"salt_seq/while/lstm_cell_30/Relu_1Relu%salt_seq/while/lstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          ╖
!salt_seq/while/lstm_cell_30/mul_2Mul)salt_seq/while/lstm_cell_30/Sigmoid_2:y:00salt_seq/while/lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          щ
3salt_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemsalt_seq_while_placeholder_1salt_seq_while_placeholder%salt_seq/while/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥V
salt_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
salt_seq/while/addAddV2salt_seq_while_placeholdersalt_seq/while/add/y:output:0*
T0*
_output_shapes
: X
salt_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
salt_seq/while/add_1AddV2*salt_seq_while_salt_seq_while_loop_countersalt_seq/while/add_1/y:output:0*
T0*
_output_shapes
: t
salt_seq/while/IdentityIdentitysalt_seq/while/add_1:z:0^salt_seq/while/NoOp*
T0*
_output_shapes
: О
salt_seq/while/Identity_1Identity0salt_seq_while_salt_seq_while_maximum_iterations^salt_seq/while/NoOp*
T0*
_output_shapes
: t
salt_seq/while/Identity_2Identitysalt_seq/while/add:z:0^salt_seq/while/NoOp*
T0*
_output_shapes
: ┤
salt_seq/while/Identity_3IdentityCsalt_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^salt_seq/while/NoOp*
T0*
_output_shapes
: :щш╥Ф
salt_seq/while/Identity_4Identity%salt_seq/while/lstm_cell_30/mul_2:z:0^salt_seq/while/NoOp*
T0*'
_output_shapes
:          Ф
salt_seq/while/Identity_5Identity%salt_seq/while/lstm_cell_30/add_1:z:0^salt_seq/while/NoOp*
T0*'
_output_shapes
:          Ї
salt_seq/while/NoOpNoOp3^salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOp2^salt_seq/while/lstm_cell_30/MatMul/ReadVariableOp4^salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
salt_seq_while_identity salt_seq/while/Identity:output:0"?
salt_seq_while_identity_1"salt_seq/while/Identity_1:output:0"?
salt_seq_while_identity_2"salt_seq/while/Identity_2:output:0"?
salt_seq_while_identity_3"salt_seq/while/Identity_3:output:0"?
salt_seq_while_identity_4"salt_seq/while/Identity_4:output:0"?
salt_seq_while_identity_5"salt_seq/while/Identity_5:output:0"|
;salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource=salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource_0"~
<salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource>salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource_0"z
:salt_seq_while_lstm_cell_30_matmul_readvariableop_resource<salt_seq_while_lstm_cell_30_matmul_readvariableop_resource_0"T
'salt_seq_while_salt_seq_strided_slice_1)salt_seq_while_salt_seq_strided_slice_1_0"╠
csalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensoresalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2h
2salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOp2salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOp2f
1salt_seq/while/lstm_cell_30/MatMul/ReadVariableOp1salt_seq/while/lstm_cell_30/MatMul/ReadVariableOp2j
3salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOp3salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
О8
Е
C__inference_qty_seq_layer_call_and_return_conditional_losses_126048

inputs&
lstm_cell_31_125966:	А&
lstm_cell_31_125968:	 А"
lstm_cell_31_125970:	А
identityИв$lstm_cell_31/StatefulPartitionedCallвwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskї
$lstm_cell_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_31_125966lstm_cell_31_125968lstm_cell_31_125970*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_125965n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╖
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_31_125966lstm_cell_31_125968lstm_cell_31_125970*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_125979*
condR
while_cond_125978*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          u
NoOpNoOp%^lstm_cell_31/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_31/StatefulPartitionedCall$lstm_cell_31/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╕F
Щ
__inference__traced_save_129461
file_prefix/
+savev2_salt_pred_kernel_read_readvariableop-
)savev2_salt_pred_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop;
7savev2_salt_seq_lstm_cell_30_kernel_read_readvariableopE
Asavev2_salt_seq_lstm_cell_30_recurrent_kernel_read_readvariableop9
5savev2_salt_seq_lstm_cell_30_bias_read_readvariableop:
6savev2_qty_seq_lstm_cell_31_kernel_read_readvariableopD
@savev2_qty_seq_lstm_cell_31_recurrent_kernel_read_readvariableop8
4savev2_qty_seq_lstm_cell_31_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_salt_pred_kernel_m_read_readvariableop4
0savev2_adam_salt_pred_bias_m_read_readvariableopB
>savev2_adam_salt_seq_lstm_cell_30_kernel_m_read_readvariableopL
Hsavev2_adam_salt_seq_lstm_cell_30_recurrent_kernel_m_read_readvariableop@
<savev2_adam_salt_seq_lstm_cell_30_bias_m_read_readvariableopA
=savev2_adam_qty_seq_lstm_cell_31_kernel_m_read_readvariableopK
Gsavev2_adam_qty_seq_lstm_cell_31_recurrent_kernel_m_read_readvariableop?
;savev2_adam_qty_seq_lstm_cell_31_bias_m_read_readvariableop6
2savev2_adam_salt_pred_kernel_v_read_readvariableop4
0savev2_adam_salt_pred_bias_v_read_readvariableopB
>savev2_adam_salt_seq_lstm_cell_30_kernel_v_read_readvariableopL
Hsavev2_adam_salt_seq_lstm_cell_30_recurrent_kernel_v_read_readvariableop@
<savev2_adam_salt_seq_lstm_cell_30_bias_v_read_readvariableopA
=savev2_adam_qty_seq_lstm_cell_31_kernel_v_read_readvariableopK
Gsavev2_adam_qty_seq_lstm_cell_31_recurrent_kernel_v_read_readvariableop?
;savev2_adam_qty_seq_lstm_cell_31_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ╡
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*▐
value╘B╤ B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHн
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B А
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_salt_pred_kernel_read_readvariableop)savev2_salt_pred_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop7savev2_salt_seq_lstm_cell_30_kernel_read_readvariableopAsavev2_salt_seq_lstm_cell_30_recurrent_kernel_read_readvariableop5savev2_salt_seq_lstm_cell_30_bias_read_readvariableop6savev2_qty_seq_lstm_cell_31_kernel_read_readvariableop@savev2_qty_seq_lstm_cell_31_recurrent_kernel_read_readvariableop4savev2_qty_seq_lstm_cell_31_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_salt_pred_kernel_m_read_readvariableop0savev2_adam_salt_pred_bias_m_read_readvariableop>savev2_adam_salt_seq_lstm_cell_30_kernel_m_read_readvariableopHsavev2_adam_salt_seq_lstm_cell_30_recurrent_kernel_m_read_readvariableop<savev2_adam_salt_seq_lstm_cell_30_bias_m_read_readvariableop=savev2_adam_qty_seq_lstm_cell_31_kernel_m_read_readvariableopGsavev2_adam_qty_seq_lstm_cell_31_recurrent_kernel_m_read_readvariableop;savev2_adam_qty_seq_lstm_cell_31_bias_m_read_readvariableop2savev2_adam_salt_pred_kernel_v_read_readvariableop0savev2_adam_salt_pred_bias_v_read_readvariableop>savev2_adam_salt_seq_lstm_cell_30_kernel_v_read_readvariableopHsavev2_adam_salt_seq_lstm_cell_30_recurrent_kernel_v_read_readvariableop<savev2_adam_salt_seq_lstm_cell_30_bias_v_read_readvariableop=savev2_adam_qty_seq_lstm_cell_31_kernel_v_read_readvariableopGsavev2_adam_qty_seq_lstm_cell_31_recurrent_kernel_v_read_readvariableop;savev2_adam_qty_seq_lstm_cell_31_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Е
_input_shapesє
Ё: :@:: : : : : :	А:	 А:А:	А:	 А:А: : :@::	А:	 А:А:	А:	 А:А:@::	А:	 А:А:	А:	 А:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	А:%	!

_output_shapes
:	 А:!


_output_shapes	
:А:%!

_output_shapes
:	А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:%!

_output_shapes
:	А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:%!

_output_shapes
:	А:%!

_output_shapes
:	 А:!

_output_shapes	
:А: 

_output_shapes
: 
П
╖
(__inference_qty_seq_layer_call_fn_128468
inputs_0
unknown:	А
	unknown_0:	 А
	unknown_1:	А
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_126239o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
у
Ч
"__inference__traced_restore_129564
file_prefix3
!assignvariableop_salt_pred_kernel:@/
!assignvariableop_1_salt_pred_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: B
/assignvariableop_7_salt_seq_lstm_cell_30_kernel:	АL
9assignvariableop_8_salt_seq_lstm_cell_30_recurrent_kernel:	 А<
-assignvariableop_9_salt_seq_lstm_cell_30_bias:	АB
/assignvariableop_10_qty_seq_lstm_cell_31_kernel:	АL
9assignvariableop_11_qty_seq_lstm_cell_31_recurrent_kernel:	 А<
-assignvariableop_12_qty_seq_lstm_cell_31_bias:	А#
assignvariableop_13_total: #
assignvariableop_14_count: =
+assignvariableop_15_adam_salt_pred_kernel_m:@7
)assignvariableop_16_adam_salt_pred_bias_m:J
7assignvariableop_17_adam_salt_seq_lstm_cell_30_kernel_m:	АT
Aassignvariableop_18_adam_salt_seq_lstm_cell_30_recurrent_kernel_m:	 АD
5assignvariableop_19_adam_salt_seq_lstm_cell_30_bias_m:	АI
6assignvariableop_20_adam_qty_seq_lstm_cell_31_kernel_m:	АS
@assignvariableop_21_adam_qty_seq_lstm_cell_31_recurrent_kernel_m:	 АC
4assignvariableop_22_adam_qty_seq_lstm_cell_31_bias_m:	А=
+assignvariableop_23_adam_salt_pred_kernel_v:@7
)assignvariableop_24_adam_salt_pred_bias_v:J
7assignvariableop_25_adam_salt_seq_lstm_cell_30_kernel_v:	АT
Aassignvariableop_26_adam_salt_seq_lstm_cell_30_recurrent_kernel_v:	 АD
5assignvariableop_27_adam_salt_seq_lstm_cell_30_bias_v:	АI
6assignvariableop_28_adam_qty_seq_lstm_cell_31_kernel_v:	АS
@assignvariableop_29_adam_qty_seq_lstm_cell_31_recurrent_kernel_v:	 АC
4assignvariableop_30_adam_qty_seq_lstm_cell_31_bias_v:	А
identity_32ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╕
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*▐
value╘B╤ B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH░
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ┴
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOpAssignVariableOp!assignvariableop_salt_pred_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_1AssignVariableOp!assignvariableop_1_salt_pred_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_7AssignVariableOp/assignvariableop_7_salt_seq_lstm_cell_30_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_8AssignVariableOp9assignvariableop_8_salt_seq_lstm_cell_30_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_9AssignVariableOp-assignvariableop_9_salt_seq_lstm_cell_30_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_10AssignVariableOp/assignvariableop_10_qty_seq_lstm_cell_31_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_11AssignVariableOp9assignvariableop_11_qty_seq_lstm_cell_31_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_12AssignVariableOp-assignvariableop_12_qty_seq_lstm_cell_31_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_salt_pred_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_salt_pred_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adam_salt_seq_lstm_cell_30_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_18AssignVariableOpAassignvariableop_18_adam_salt_seq_lstm_cell_30_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_19AssignVariableOp5assignvariableop_19_adam_salt_seq_lstm_cell_30_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_qty_seq_lstm_cell_31_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_qty_seq_lstm_cell_31_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_qty_seq_lstm_cell_31_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_salt_pred_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_salt_pred_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_salt_seq_lstm_cell_30_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_26AssignVariableOpAassignvariableop_26_adam_salt_seq_lstm_cell_30_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_27AssignVariableOp5assignvariableop_27_adam_salt_seq_lstm_cell_30_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_qty_seq_lstm_cell_31_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_qty_seq_lstm_cell_31_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_qty_seq_lstm_cell_31_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ∙
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ц
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ы
Ў
-__inference_lstm_cell_31_layer_call_fn_129263

inputs
states_0
states_1
unknown:	А
	unknown_0:	 А
	unknown_1:	А
identity

identity_1

identity_2ИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_125965o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
°I
░
 model_salt_seq_while_body_125454:
6model_salt_seq_while_model_salt_seq_while_loop_counter@
<model_salt_seq_while_model_salt_seq_while_maximum_iterations$
 model_salt_seq_while_placeholder&
"model_salt_seq_while_placeholder_1&
"model_salt_seq_while_placeholder_2&
"model_salt_seq_while_placeholder_39
5model_salt_seq_while_model_salt_seq_strided_slice_1_0u
qmodel_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_salt_seq_tensorarrayunstack_tensorlistfromtensor_0U
Bmodel_salt_seq_while_lstm_cell_30_matmul_readvariableop_resource_0:	АW
Dmodel_salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource_0:	 АR
Cmodel_salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource_0:	А!
model_salt_seq_while_identity#
model_salt_seq_while_identity_1#
model_salt_seq_while_identity_2#
model_salt_seq_while_identity_3#
model_salt_seq_while_identity_4#
model_salt_seq_while_identity_57
3model_salt_seq_while_model_salt_seq_strided_slice_1s
omodel_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_salt_seq_tensorarrayunstack_tensorlistfromtensorS
@model_salt_seq_while_lstm_cell_30_matmul_readvariableop_resource:	АU
Bmodel_salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource:	 АP
Amodel_salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource:	АИв8model/salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOpв7model/salt_seq/while/lstm_cell_30/MatMul/ReadVariableOpв9model/salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOpЧ
Fmodel/salt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ё
8model/salt_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqmodel_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_salt_seq_tensorarrayunstack_tensorlistfromtensor_0 model_salt_seq_while_placeholderOmodel/salt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0╗
7model/salt_seq/while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOpBmodel_salt_seq_while_lstm_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0ч
(model/salt_seq/while/lstm_cell_30/MatMulMatMul?model/salt_seq/while/TensorArrayV2Read/TensorListGetItem:item:0?model/salt_seq/while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А┐
9model/salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOpDmodel_salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0╬
*model/salt_seq/while/lstm_cell_30/MatMul_1MatMul"model_salt_seq_while_placeholder_2Amodel/salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╦
%model/salt_seq/while/lstm_cell_30/addAddV22model/salt_seq/while/lstm_cell_30/MatMul:product:04model/salt_seq/while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         А╣
8model/salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOpCmodel_salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0╘
)model/salt_seq/while/lstm_cell_30/BiasAddBiasAdd)model/salt_seq/while/lstm_cell_30/add:z:0@model/salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
1model/salt_seq/while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ь
'model/salt_seq/while/lstm_cell_30/splitSplit:model/salt_seq/while/lstm_cell_30/split/split_dim:output:02model/salt_seq/while/lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitШ
)model/salt_seq/while/lstm_cell_30/SigmoidSigmoid0model/salt_seq/while/lstm_cell_30/split:output:0*
T0*'
_output_shapes
:          Ъ
+model/salt_seq/while/lstm_cell_30/Sigmoid_1Sigmoid0model/salt_seq/while/lstm_cell_30/split:output:1*
T0*'
_output_shapes
:          │
%model/salt_seq/while/lstm_cell_30/mulMul/model/salt_seq/while/lstm_cell_30/Sigmoid_1:y:0"model_salt_seq_while_placeholder_3*
T0*'
_output_shapes
:          Т
&model/salt_seq/while/lstm_cell_30/ReluRelu0model/salt_seq/while/lstm_cell_30/split:output:2*
T0*'
_output_shapes
:          ┼
'model/salt_seq/while/lstm_cell_30/mul_1Mul-model/salt_seq/while/lstm_cell_30/Sigmoid:y:04model/salt_seq/while/lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          ║
'model/salt_seq/while/lstm_cell_30/add_1AddV2)model/salt_seq/while/lstm_cell_30/mul:z:0+model/salt_seq/while/lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          Ъ
+model/salt_seq/while/lstm_cell_30/Sigmoid_2Sigmoid0model/salt_seq/while/lstm_cell_30/split:output:3*
T0*'
_output_shapes
:          П
(model/salt_seq/while/lstm_cell_30/Relu_1Relu+model/salt_seq/while/lstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          ╔
'model/salt_seq/while/lstm_cell_30/mul_2Mul/model/salt_seq/while/lstm_cell_30/Sigmoid_2:y:06model/salt_seq/while/lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          Б
9model/salt_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"model_salt_seq_while_placeholder_1 model_salt_seq_while_placeholder+model/salt_seq/while/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥\
model/salt_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Й
model/salt_seq/while/addAddV2 model_salt_seq_while_placeholder#model/salt_seq/while/add/y:output:0*
T0*
_output_shapes
: ^
model/salt_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :г
model/salt_seq/while/add_1AddV26model_salt_seq_while_model_salt_seq_while_loop_counter%model/salt_seq/while/add_1/y:output:0*
T0*
_output_shapes
: Ж
model/salt_seq/while/IdentityIdentitymodel/salt_seq/while/add_1:z:0^model/salt_seq/while/NoOp*
T0*
_output_shapes
: ж
model/salt_seq/while/Identity_1Identity<model_salt_seq_while_model_salt_seq_while_maximum_iterations^model/salt_seq/while/NoOp*
T0*
_output_shapes
: Ж
model/salt_seq/while/Identity_2Identitymodel/salt_seq/while/add:z:0^model/salt_seq/while/NoOp*
T0*
_output_shapes
: ╞
model/salt_seq/while/Identity_3IdentityImodel/salt_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/salt_seq/while/NoOp*
T0*
_output_shapes
: :щш╥ж
model/salt_seq/while/Identity_4Identity+model/salt_seq/while/lstm_cell_30/mul_2:z:0^model/salt_seq/while/NoOp*
T0*'
_output_shapes
:          ж
model/salt_seq/while/Identity_5Identity+model/salt_seq/while/lstm_cell_30/add_1:z:0^model/salt_seq/while/NoOp*
T0*'
_output_shapes
:          М
model/salt_seq/while/NoOpNoOp9^model/salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOp8^model/salt_seq/while/lstm_cell_30/MatMul/ReadVariableOp:^model/salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "G
model_salt_seq_while_identity&model/salt_seq/while/Identity:output:0"K
model_salt_seq_while_identity_1(model/salt_seq/while/Identity_1:output:0"K
model_salt_seq_while_identity_2(model/salt_seq/while/Identity_2:output:0"K
model_salt_seq_while_identity_3(model/salt_seq/while/Identity_3:output:0"K
model_salt_seq_while_identity_4(model/salt_seq/while/Identity_4:output:0"K
model_salt_seq_while_identity_5(model/salt_seq/while/Identity_5:output:0"И
Amodel_salt_seq_while_lstm_cell_30_biasadd_readvariableop_resourceCmodel_salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource_0"К
Bmodel_salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resourceDmodel_salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource_0"Ж
@model_salt_seq_while_lstm_cell_30_matmul_readvariableop_resourceBmodel_salt_seq_while_lstm_cell_30_matmul_readvariableop_resource_0"l
3model_salt_seq_while_model_salt_seq_strided_slice_15model_salt_seq_while_model_salt_seq_strided_slice_1_0"ф
omodel_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_salt_seq_tensorarrayunstack_tensorlistfromtensorqmodel_salt_seq_while_tensorarrayv2read_tensorlistgetitem_model_salt_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2t
8model/salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOp8model/salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOp2r
7model/salt_seq/while/lstm_cell_30/MatMul/ReadVariableOp7model/salt_seq/while/lstm_cell_30/MatMul/ReadVariableOp2v
9model/salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOp9model/salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
Я8
╨
while_body_126751
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_30_matmul_readvariableop_resource_0:	АH
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:	 АC
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_30_matmul_readvariableop_resource:	АF
3while_lstm_cell_30_matmul_1_readvariableop_resource:	 АA
2while_lstm_cell_30_biasadd_readvariableop_resource:	АИв)while/lstm_cell_30/BiasAdd/ReadVariableOpв(while/lstm_cell_30/MatMul/ReadVariableOpв*while/lstm_cell_30/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Э
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0║
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0б
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АЫ
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0з
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitz
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*'
_output_shapes
:          |
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*'
_output_shapes
:          Ж
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          t
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*'
_output_shapes
:          Ш
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          Н
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          |
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*'
_output_shapes
:          q
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          Ь
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥y
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          y
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          ╨

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
▐J
Ю
D__inference_salt_seq_layer_call_and_return_conditional_losses_128017
inputs_0>
+lstm_cell_30_matmul_readvariableop_resource:	А@
-lstm_cell_30_matmul_1_readvariableop_resource:	 А;
,lstm_cell_30_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_30/BiasAdd/ReadVariableOpв"lstm_cell_30/MatMul/ReadVariableOpв$lstm_cell_30/MatMul_1/ReadVariableOpвwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskП
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АМ
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АН
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А^
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▌
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitn
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*'
_output_shapes
:          p
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*'
_output_shapes
:          w
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          h
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*'
_output_shapes
:          Ж
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          {
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          p
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*'
_output_shapes
:          e
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          К
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_127933*
condR
while_cond_127932*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          └
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
С
╕
)__inference_salt_seq_layer_call_fn_127841
inputs_0
unknown:	А
	unknown_0:	 А
	unknown_1:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_125698o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
а
T
(__inference_pattern_layer_call_fn_129122
inputs_0
inputs_1
identity╗
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_126578`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:          :          :Q M
'
_output_shapes
:          
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:          
"
_user_specified_name
inputs/1
╒
Д
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_125615

inputs

states
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates
Ц

у
qty_seq_while_cond_127265,
(qty_seq_while_qty_seq_while_loop_counter2
.qty_seq_while_qty_seq_while_maximum_iterations
qty_seq_while_placeholder
qty_seq_while_placeholder_1
qty_seq_while_placeholder_2
qty_seq_while_placeholder_3.
*qty_seq_while_less_qty_seq_strided_slice_1D
@qty_seq_while_qty_seq_while_cond_127265___redundant_placeholder0D
@qty_seq_while_qty_seq_while_cond_127265___redundant_placeholder1D
@qty_seq_while_qty_seq_while_cond_127265___redundant_placeholder2D
@qty_seq_while_qty_seq_while_cond_127265___redundant_placeholder3
qty_seq_while_identity
В
qty_seq/while/LessLessqty_seq_while_placeholder*qty_seq_while_less_qty_seq_strided_slice_1*
T0*
_output_shapes
: [
qty_seq/while/IdentityIdentityqty_seq/while/Less:z:0*
T0
*
_output_shapes
: "9
qty_seq_while_identityqty_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
┌
я
 model_salt_seq_while_cond_125453:
6model_salt_seq_while_model_salt_seq_while_loop_counter@
<model_salt_seq_while_model_salt_seq_while_maximum_iterations$
 model_salt_seq_while_placeholder&
"model_salt_seq_while_placeholder_1&
"model_salt_seq_while_placeholder_2&
"model_salt_seq_while_placeholder_3<
8model_salt_seq_while_less_model_salt_seq_strided_slice_1R
Nmodel_salt_seq_while_model_salt_seq_while_cond_125453___redundant_placeholder0R
Nmodel_salt_seq_while_model_salt_seq_while_cond_125453___redundant_placeholder1R
Nmodel_salt_seq_while_model_salt_seq_while_cond_125453___redundant_placeholder2R
Nmodel_salt_seq_while_model_salt_seq_while_cond_125453___redundant_placeholder3!
model_salt_seq_while_identity
Ю
model/salt_seq/while/LessLess model_salt_seq_while_placeholder8model_salt_seq_while_less_model_salt_seq_strided_slice_1*
T0*
_output_shapes
: i
model/salt_seq/while/IdentityIdentitymodel/salt_seq/while/Less:z:0*
T0
*
_output_shapes
: "G
model_salt_seq_while_identity&model/salt_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
 
▒
A__inference_model_layer_call_and_return_conditional_losses_127061

inputs
inputs_1!
qty_seq_127038:	А!
qty_seq_127040:	 А
qty_seq_127042:	А"
salt_seq_127045:	А"
salt_seq_127047:	 А
salt_seq_127049:	А"
salt_pred_127055:@
salt_pred_127057:
identityИвqty_seq/StatefulPartitionedCallв!qty_seq_2/StatefulPartitionedCallв!salt_pred/StatefulPartitionedCallв salt_seq/StatefulPartitionedCallв"salt_seq_2/StatefulPartitionedCallА
qty_seq/StatefulPartitionedCallStatefulPartitionedCallinputs_1qty_seq_127038qty_seq_127040qty_seq_127042*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_127000Г
 salt_seq/StatefulPartitionedCallStatefulPartitionedCallinputssalt_seq_127045salt_seq_127047salt_seq_127049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_126835я
"salt_seq_2/StatefulPartitionedCallStatefulPartitionedCall)salt_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_126676С
!qty_seq_2/StatefulPartitionedCallStatefulPartitionedCall(qty_seq/StatefulPartitionedCall:output:0#^salt_seq_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_126653И
pattern/PartitionedCallPartitionedCall+salt_seq_2/StatefulPartitionedCall:output:0*qty_seq_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_126578О
!salt_pred/StatefulPartitionedCallStatefulPartitionedCall pattern/PartitionedCall:output:0salt_pred_127055salt_pred_127057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_126590y
IdentityIdentity*salt_pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         °
NoOpNoOp ^qty_seq/StatefulPartitionedCall"^qty_seq_2/StatefulPartitionedCall"^salt_pred/StatefulPartitionedCall!^salt_seq/StatefulPartitionedCall#^salt_seq_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         :         : : : : : : : : 2B
qty_seq/StatefulPartitionedCallqty_seq/StatefulPartitionedCall2F
!qty_seq_2/StatefulPartitionedCall!qty_seq_2/StatefulPartitionedCall2F
!salt_pred/StatefulPartitionedCall!salt_pred/StatefulPartitionedCall2D
 salt_seq/StatefulPartitionedCall salt_seq/StatefulPartitionedCall2H
"salt_seq_2/StatefulPartitionedCall"salt_seq_2/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:SO
+
_output_shapes
:         
 
_user_specified_nameinputs
▓

ў
salt_seq_while_cond_127697.
*salt_seq_while_salt_seq_while_loop_counter4
0salt_seq_while_salt_seq_while_maximum_iterations
salt_seq_while_placeholder 
salt_seq_while_placeholder_1 
salt_seq_while_placeholder_2 
salt_seq_while_placeholder_30
,salt_seq_while_less_salt_seq_strided_slice_1F
Bsalt_seq_while_salt_seq_while_cond_127697___redundant_placeholder0F
Bsalt_seq_while_salt_seq_while_cond_127697___redundant_placeholder1F
Bsalt_seq_while_salt_seq_while_cond_127697___redundant_placeholder2F
Bsalt_seq_while_salt_seq_while_cond_127697___redundant_placeholder3
salt_seq_while_identity
Ж
salt_seq/while/LessLesssalt_seq_while_placeholder,salt_seq_while_less_salt_seq_strided_slice_1*
T0*
_output_shapes
: ]
salt_seq/while/IdentityIdentitysalt_seq/while/Less:z:0*
T0
*
_output_shapes
: ";
salt_seq_while_identity salt_seq/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╚	
Ў
E__inference_salt_pred_layer_call_and_return_conditional_losses_126590

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╚	
Ў
E__inference_salt_pred_layer_call_and_return_conditional_losses_129148

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
▌
Ж
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_129246

inputs
states_0
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
ё
c
*__inference_qty_seq_2_layer_call_fn_129099

inputs
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_126653o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ў
╡
(__inference_qty_seq_layer_call_fn_128479

inputs
unknown:	А
	unknown_0:	 А
	unknown_1:	А
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_126399o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Я8
╨
while_body_128978
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_31_matmul_readvariableop_resource_0:	АH
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:	 АC
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_31_matmul_readvariableop_resource:	АF
3while_lstm_cell_31_matmul_1_readvariableop_resource:	 АA
2while_lstm_cell_31_biasadd_readvariableop_resource:	АИв)while/lstm_cell_31/BiasAdd/ReadVariableOpв(while/lstm_cell_31/MatMul/ReadVariableOpв*while/lstm_cell_31/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Э
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0║
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0б
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АЫ
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0з
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitz
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*'
_output_shapes
:          |
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*'
_output_shapes
:          Ж
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          t
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ш
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          Н
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          |
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*'
_output_shapes
:          q
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          Ь
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥y
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          y
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          ╨

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
╪A
╨

qty_seq_while_body_127559,
(qty_seq_while_qty_seq_while_loop_counter2
.qty_seq_while_qty_seq_while_maximum_iterations
qty_seq_while_placeholder
qty_seq_while_placeholder_1
qty_seq_while_placeholder_2
qty_seq_while_placeholder_3+
'qty_seq_while_qty_seq_strided_slice_1_0g
cqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0N
;qty_seq_while_lstm_cell_31_matmul_readvariableop_resource_0:	АP
=qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource_0:	 АK
<qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource_0:	А
qty_seq_while_identity
qty_seq_while_identity_1
qty_seq_while_identity_2
qty_seq_while_identity_3
qty_seq_while_identity_4
qty_seq_while_identity_5)
%qty_seq_while_qty_seq_strided_slice_1e
aqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensorL
9qty_seq_while_lstm_cell_31_matmul_readvariableop_resource:	АN
;qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource:	 АI
:qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource:	АИв1qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOpв0qty_seq/while/lstm_cell_31/MatMul/ReadVariableOpв2qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOpР
?qty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1qty_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemcqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0qty_seq_while_placeholderHqty_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0н
0qty_seq/while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp;qty_seq_while_lstm_cell_31_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0╥
!qty_seq/while/lstm_cell_31/MatMulMatMul8qty_seq/while/TensorArrayV2Read/TensorListGetItem:item:08qty_seq/while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А▒
2qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp=qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0╣
#qty_seq/while/lstm_cell_31/MatMul_1MatMulqty_seq_while_placeholder_2:qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╢
qty_seq/while/lstm_cell_31/addAddV2+qty_seq/while/lstm_cell_31/MatMul:product:0-qty_seq/while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         Ал
1qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp<qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0┐
"qty_seq/while/lstm_cell_31/BiasAddBiasAdd"qty_seq/while/lstm_cell_31/add:z:09qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аl
*qty_seq/while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :З
 qty_seq/while/lstm_cell_31/splitSplit3qty_seq/while/lstm_cell_31/split/split_dim:output:0+qty_seq/while/lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitК
"qty_seq/while/lstm_cell_31/SigmoidSigmoid)qty_seq/while/lstm_cell_31/split:output:0*
T0*'
_output_shapes
:          М
$qty_seq/while/lstm_cell_31/Sigmoid_1Sigmoid)qty_seq/while/lstm_cell_31/split:output:1*
T0*'
_output_shapes
:          Ю
qty_seq/while/lstm_cell_31/mulMul(qty_seq/while/lstm_cell_31/Sigmoid_1:y:0qty_seq_while_placeholder_3*
T0*'
_output_shapes
:          Д
qty_seq/while/lstm_cell_31/ReluRelu)qty_seq/while/lstm_cell_31/split:output:2*
T0*'
_output_shapes
:          ░
 qty_seq/while/lstm_cell_31/mul_1Mul&qty_seq/while/lstm_cell_31/Sigmoid:y:0-qty_seq/while/lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          е
 qty_seq/while/lstm_cell_31/add_1AddV2"qty_seq/while/lstm_cell_31/mul:z:0$qty_seq/while/lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          М
$qty_seq/while/lstm_cell_31/Sigmoid_2Sigmoid)qty_seq/while/lstm_cell_31/split:output:3*
T0*'
_output_shapes
:          Б
!qty_seq/while/lstm_cell_31/Relu_1Relu$qty_seq/while/lstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          ┤
 qty_seq/while/lstm_cell_31/mul_2Mul(qty_seq/while/lstm_cell_31/Sigmoid_2:y:0/qty_seq/while/lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          х
2qty_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemqty_seq_while_placeholder_1qty_seq_while_placeholder$qty_seq/while/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥U
qty_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
qty_seq/while/addAddV2qty_seq_while_placeholderqty_seq/while/add/y:output:0*
T0*
_output_shapes
: W
qty_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
qty_seq/while/add_1AddV2(qty_seq_while_qty_seq_while_loop_counterqty_seq/while/add_1/y:output:0*
T0*
_output_shapes
: q
qty_seq/while/IdentityIdentityqty_seq/while/add_1:z:0^qty_seq/while/NoOp*
T0*
_output_shapes
: К
qty_seq/while/Identity_1Identity.qty_seq_while_qty_seq_while_maximum_iterations^qty_seq/while/NoOp*
T0*
_output_shapes
: q
qty_seq/while/Identity_2Identityqty_seq/while/add:z:0^qty_seq/while/NoOp*
T0*
_output_shapes
: ▒
qty_seq/while/Identity_3IdentityBqty_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^qty_seq/while/NoOp*
T0*
_output_shapes
: :щш╥С
qty_seq/while/Identity_4Identity$qty_seq/while/lstm_cell_31/mul_2:z:0^qty_seq/while/NoOp*
T0*'
_output_shapes
:          С
qty_seq/while/Identity_5Identity$qty_seq/while/lstm_cell_31/add_1:z:0^qty_seq/while/NoOp*
T0*'
_output_shapes
:          Ё
qty_seq/while/NoOpNoOp2^qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOp1^qty_seq/while/lstm_cell_31/MatMul/ReadVariableOp3^qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
qty_seq_while_identityqty_seq/while/Identity:output:0"=
qty_seq_while_identity_1!qty_seq/while/Identity_1:output:0"=
qty_seq_while_identity_2!qty_seq/while/Identity_2:output:0"=
qty_seq_while_identity_3!qty_seq/while/Identity_3:output:0"=
qty_seq_while_identity_4!qty_seq/while/Identity_4:output:0"=
qty_seq_while_identity_5!qty_seq/while/Identity_5:output:0"z
:qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource<qty_seq_while_lstm_cell_31_biasadd_readvariableop_resource_0"|
;qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource=qty_seq_while_lstm_cell_31_matmul_1_readvariableop_resource_0"x
9qty_seq_while_lstm_cell_31_matmul_readvariableop_resource;qty_seq_while_lstm_cell_31_matmul_readvariableop_resource_0"P
%qty_seq_while_qty_seq_strided_slice_1'qty_seq_while_qty_seq_strided_slice_1_0"╚
aqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensorcqty_seq_while_tensorarrayv2read_tensorlistgetitem_qty_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2f
1qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOp1qty_seq/while/lstm_cell_31/BiasAdd/ReadVariableOp2d
0qty_seq/while/lstm_cell_31/MatMul/ReadVariableOp0qty_seq/while/lstm_cell_31/MatMul/ReadVariableOp2h
2qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOp2qty_seq/while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
∙"
у
while_body_125979
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_31_126003_0:	А.
while_lstm_cell_31_126005_0:	 А*
while_lstm_cell_31_126007_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_31_126003:	А,
while_lstm_cell_31_126005:	 А(
while_lstm_cell_31_126007:	АИв*while/lstm_cell_31/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0│
*while/lstm_cell_31/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_31_126003_0while_lstm_cell_31_126005_0while_lstm_cell_31_126007_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_125965▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_31/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥Р
while/Identity_4Identity3while/lstm_cell_31/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          Р
while/Identity_5Identity3while/lstm_cell_31/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          y

while/NoOpNoOp+^while/lstm_cell_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_31_126003while_lstm_cell_31_126003_0"8
while_lstm_cell_31_126005while_lstm_cell_31_126005_0"8
while_lstm_cell_31_126007while_lstm_cell_31_126007_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2X
*while/lstm_cell_31/StatefulPartitionedCall*while/lstm_cell_31/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
╒┴
У
A__inference_model_layer_call_and_return_conditional_losses_127806
inputs_0
inputs_1F
3qty_seq_lstm_cell_31_matmul_readvariableop_resource:	АH
5qty_seq_lstm_cell_31_matmul_1_readvariableop_resource:	 АC
4qty_seq_lstm_cell_31_biasadd_readvariableop_resource:	АG
4salt_seq_lstm_cell_30_matmul_readvariableop_resource:	АI
6salt_seq_lstm_cell_30_matmul_1_readvariableop_resource:	 АD
5salt_seq_lstm_cell_30_biasadd_readvariableop_resource:	А:
(salt_pred_matmul_readvariableop_resource:@7
)salt_pred_biasadd_readvariableop_resource:
identityИв+qty_seq/lstm_cell_31/BiasAdd/ReadVariableOpв*qty_seq/lstm_cell_31/MatMul/ReadVariableOpв,qty_seq/lstm_cell_31/MatMul_1/ReadVariableOpвqty_seq/whileв salt_pred/BiasAdd/ReadVariableOpвsalt_pred/MatMul/ReadVariableOpв,salt_seq/lstm_cell_30/BiasAdd/ReadVariableOpв+salt_seq/lstm_cell_30/MatMul/ReadVariableOpв-salt_seq/lstm_cell_30/MatMul_1/ReadVariableOpвsalt_seq/whileE
qty_seq/ShapeShapeinputs_1*
T0*
_output_shapes
:e
qty_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
qty_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
qty_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
qty_seq/strided_sliceStridedSliceqty_seq/Shape:output:0$qty_seq/strided_slice/stack:output:0&qty_seq/strided_slice/stack_1:output:0&qty_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
qty_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Л
qty_seq/zeros/packedPackqty_seq/strided_slice:output:0qty_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
qty_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Д
qty_seq/zerosFillqty_seq/zeros/packed:output:0qty_seq/zeros/Const:output:0*
T0*'
_output_shapes
:          Z
qty_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : П
qty_seq/zeros_1/packedPackqty_seq/strided_slice:output:0!qty_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
qty_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    К
qty_seq/zeros_1Fillqty_seq/zeros_1/packed:output:0qty_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:          k
qty_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
qty_seq/transpose	Transposeinputs_1qty_seq/transpose/perm:output:0*
T0*+
_output_shapes
:         T
qty_seq/Shape_1Shapeqty_seq/transpose:y:0*
T0*
_output_shapes
:g
qty_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
qty_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
qty_seq/strided_slice_1StridedSliceqty_seq/Shape_1:output:0&qty_seq/strided_slice_1/stack:output:0(qty_seq/strided_slice_1/stack_1:output:0(qty_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#qty_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
qty_seq/TensorArrayV2TensorListReserve,qty_seq/TensorArrayV2/element_shape:output:0 qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥О
=qty_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       °
/qty_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorqty_seq/transpose:y:0Fqty_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥g
qty_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
qty_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:С
qty_seq/strided_slice_2StridedSliceqty_seq/transpose:y:0&qty_seq/strided_slice_2/stack:output:0(qty_seq/strided_slice_2/stack_1:output:0(qty_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЯ
*qty_seq/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3qty_seq_lstm_cell_31_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0о
qty_seq/lstm_cell_31/MatMulMatMul qty_seq/strided_slice_2:output:02qty_seq/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аг
,qty_seq/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5qty_seq_lstm_cell_31_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0и
qty_seq/lstm_cell_31/MatMul_1MatMulqty_seq/zeros:output:04qty_seq/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ад
qty_seq/lstm_cell_31/addAddV2%qty_seq/lstm_cell_31/MatMul:product:0'qty_seq/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АЭ
+qty_seq/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4qty_seq_lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0н
qty_seq/lstm_cell_31/BiasAddBiasAddqty_seq/lstm_cell_31/add:z:03qty_seq/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аf
$qty_seq/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
qty_seq/lstm_cell_31/splitSplit-qty_seq/lstm_cell_31/split/split_dim:output:0%qty_seq/lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split~
qty_seq/lstm_cell_31/SigmoidSigmoid#qty_seq/lstm_cell_31/split:output:0*
T0*'
_output_shapes
:          А
qty_seq/lstm_cell_31/Sigmoid_1Sigmoid#qty_seq/lstm_cell_31/split:output:1*
T0*'
_output_shapes
:          П
qty_seq/lstm_cell_31/mulMul"qty_seq/lstm_cell_31/Sigmoid_1:y:0qty_seq/zeros_1:output:0*
T0*'
_output_shapes
:          x
qty_seq/lstm_cell_31/ReluRelu#qty_seq/lstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ю
qty_seq/lstm_cell_31/mul_1Mul qty_seq/lstm_cell_31/Sigmoid:y:0'qty_seq/lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          У
qty_seq/lstm_cell_31/add_1AddV2qty_seq/lstm_cell_31/mul:z:0qty_seq/lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          А
qty_seq/lstm_cell_31/Sigmoid_2Sigmoid#qty_seq/lstm_cell_31/split:output:3*
T0*'
_output_shapes
:          u
qty_seq/lstm_cell_31/Relu_1Reluqty_seq/lstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          в
qty_seq/lstm_cell_31/mul_2Mul"qty_seq/lstm_cell_31/Sigmoid_2:y:0)qty_seq/lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          v
%qty_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╨
qty_seq/TensorArrayV2_1TensorListReserve.qty_seq/TensorArrayV2_1/element_shape:output:0 qty_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥N
qty_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 qty_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
qty_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Є
qty_seq/whileWhile#qty_seq/while/loop_counter:output:0)qty_seq/while/maximum_iterations:output:0qty_seq/time:output:0 qty_seq/TensorArrayV2_1:handle:0qty_seq/zeros:output:0qty_seq/zeros_1:output:0 qty_seq/strided_slice_1:output:0?qty_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:03qty_seq_lstm_cell_31_matmul_readvariableop_resource5qty_seq_lstm_cell_31_matmul_1_readvariableop_resource4qty_seq_lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
qty_seq_while_body_127559*%
condR
qty_seq_while_cond_127558*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Й
8qty_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┌
*qty_seq/TensorArrayV2Stack/TensorListStackTensorListStackqty_seq/while:output:3Aqty_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0p
qty_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
qty_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
qty_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
qty_seq/strided_slice_3StridedSlice3qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0&qty_seq/strided_slice_3/stack:output:0(qty_seq/strided_slice_3/stack_1:output:0(qty_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskm
qty_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          о
qty_seq/transpose_1	Transpose3qty_seq/TensorArrayV2Stack/TensorListStack:tensor:0!qty_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:          c
qty_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    F
salt_seq/ShapeShapeinputs_0*
T0*
_output_shapes
:f
salt_seq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
salt_seq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
salt_seq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
salt_seq/strided_sliceStridedSlicesalt_seq/Shape:output:0%salt_seq/strided_slice/stack:output:0'salt_seq/strided_slice/stack_1:output:0'salt_seq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
salt_seq/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : О
salt_seq/zeros/packedPacksalt_seq/strided_slice:output:0 salt_seq/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
salt_seq/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    З
salt_seq/zerosFillsalt_seq/zeros/packed:output:0salt_seq/zeros/Const:output:0*
T0*'
_output_shapes
:          [
salt_seq/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Т
salt_seq/zeros_1/packedPacksalt_seq/strided_slice:output:0"salt_seq/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
salt_seq/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Н
salt_seq/zeros_1Fill salt_seq/zeros_1/packed:output:0salt_seq/zeros_1/Const:output:0*
T0*'
_output_shapes
:          l
salt_seq/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Б
salt_seq/transpose	Transposeinputs_0 salt_seq/transpose/perm:output:0*
T0*+
_output_shapes
:         V
salt_seq/Shape_1Shapesalt_seq/transpose:y:0*
T0*
_output_shapes
:h
salt_seq/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 salt_seq/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
salt_seq/strided_slice_1StridedSlicesalt_seq/Shape_1:output:0'salt_seq/strided_slice_1/stack:output:0)salt_seq/strided_slice_1/stack_1:output:0)salt_seq/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$salt_seq/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╧
salt_seq/TensorArrayV2TensorListReserve-salt_seq/TensorArrayV2/element_shape:output:0!salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥П
>salt_seq/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       √
0salt_seq/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsalt_seq/transpose:y:0Gsalt_seq/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥h
salt_seq/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 salt_seq/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ц
salt_seq/strided_slice_2StridedSlicesalt_seq/transpose:y:0'salt_seq/strided_slice_2/stack:output:0)salt_seq/strided_slice_2/stack_1:output:0)salt_seq/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskб
+salt_seq/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp4salt_seq_lstm_cell_30_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0▒
salt_seq/lstm_cell_30/MatMulMatMul!salt_seq/strided_slice_2:output:03salt_seq/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
-salt_seq/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp6salt_seq_lstm_cell_30_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0л
salt_seq/lstm_cell_30/MatMul_1MatMulsalt_seq/zeros:output:05salt_seq/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
salt_seq/lstm_cell_30/addAddV2&salt_seq/lstm_cell_30/MatMul:product:0(salt_seq/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АЯ
,salt_seq/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp5salt_seq_lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
salt_seq/lstm_cell_30/BiasAddBiasAddsalt_seq/lstm_cell_30/add:z:04salt_seq/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аg
%salt_seq/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :°
salt_seq/lstm_cell_30/splitSplit.salt_seq/lstm_cell_30/split/split_dim:output:0&salt_seq/lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitА
salt_seq/lstm_cell_30/SigmoidSigmoid$salt_seq/lstm_cell_30/split:output:0*
T0*'
_output_shapes
:          В
salt_seq/lstm_cell_30/Sigmoid_1Sigmoid$salt_seq/lstm_cell_30/split:output:1*
T0*'
_output_shapes
:          Т
salt_seq/lstm_cell_30/mulMul#salt_seq/lstm_cell_30/Sigmoid_1:y:0salt_seq/zeros_1:output:0*
T0*'
_output_shapes
:          z
salt_seq/lstm_cell_30/ReluRelu$salt_seq/lstm_cell_30/split:output:2*
T0*'
_output_shapes
:          б
salt_seq/lstm_cell_30/mul_1Mul!salt_seq/lstm_cell_30/Sigmoid:y:0(salt_seq/lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          Ц
salt_seq/lstm_cell_30/add_1AddV2salt_seq/lstm_cell_30/mul:z:0salt_seq/lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          В
salt_seq/lstm_cell_30/Sigmoid_2Sigmoid$salt_seq/lstm_cell_30/split:output:3*
T0*'
_output_shapes
:          w
salt_seq/lstm_cell_30/Relu_1Relusalt_seq/lstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          е
salt_seq/lstm_cell_30/mul_2Mul#salt_seq/lstm_cell_30/Sigmoid_2:y:0*salt_seq/lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          w
&salt_seq/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╙
salt_seq/TensorArrayV2_1TensorListReserve/salt_seq/TensorArrayV2_1/element_shape:output:0!salt_seq/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥O
salt_seq/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!salt_seq/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         ]
salt_seq/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : А
salt_seq/whileWhile$salt_seq/while/loop_counter:output:0*salt_seq/while/maximum_iterations:output:0salt_seq/time:output:0!salt_seq/TensorArrayV2_1:handle:0salt_seq/zeros:output:0salt_seq/zeros_1:output:0!salt_seq/strided_slice_1:output:0@salt_seq/TensorArrayUnstack/TensorListFromTensor:output_handle:04salt_seq_lstm_cell_30_matmul_readvariableop_resource6salt_seq_lstm_cell_30_matmul_1_readvariableop_resource5salt_seq_lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
salt_seq_while_body_127698*&
condR
salt_seq_while_cond_127697*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations К
9salt_seq/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ▌
+salt_seq/TensorArrayV2Stack/TensorListStackTensorListStacksalt_seq/while:output:3Bsalt_seq/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0q
salt_seq/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         j
 salt_seq/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 salt_seq/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
salt_seq/strided_slice_3StridedSlice4salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0'salt_seq/strided_slice_3/stack:output:0)salt_seq/strided_slice_3/stack_1:output:0)salt_seq/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maskn
salt_seq/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ▒
salt_seq/transpose_1	Transpose4salt_seq/TensorArrayV2Stack/TensorListStack:tensor:0"salt_seq/transpose_1/perm:output:0*
T0*+
_output_shapes
:          d
salt_seq/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
salt_seq_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?Х
salt_seq_2/dropout/MulMul!salt_seq/strided_slice_3:output:0!salt_seq_2/dropout/Const:output:0*
T0*'
_output_shapes
:          i
salt_seq_2/dropout/ShapeShape!salt_seq/strided_slice_3:output:0*
T0*
_output_shapes
:о
/salt_seq_2/dropout/random_uniform/RandomUniformRandomUniform!salt_seq_2/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*

seedf
!salt_seq_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>╟
salt_seq_2/dropout/GreaterEqualGreaterEqual8salt_seq_2/dropout/random_uniform/RandomUniform:output:0*salt_seq_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          Е
salt_seq_2/dropout/CastCast#salt_seq_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          К
salt_seq_2/dropout/Mul_1Mulsalt_seq_2/dropout/Mul:z:0salt_seq_2/dropout/Cast:y:0*
T0*'
_output_shapes
:          \
qty_seq_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?Т
qty_seq_2/dropout/MulMul qty_seq/strided_slice_3:output:0 qty_seq_2/dropout/Const:output:0*
T0*'
_output_shapes
:          g
qty_seq_2/dropout/ShapeShape qty_seq/strided_slice_3:output:0*
T0*
_output_shapes
:╣
.qty_seq_2/dropout/random_uniform/RandomUniformRandomUniform qty_seq_2/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*

seed*
seed2e
 qty_seq_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>─
qty_seq_2/dropout/GreaterEqualGreaterEqual7qty_seq_2/dropout/random_uniform/RandomUniform:output:0)qty_seq_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          Г
qty_seq_2/dropout/CastCast"qty_seq_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          З
qty_seq_2/dropout/Mul_1Mulqty_seq_2/dropout/Mul:z:0qty_seq_2/dropout/Cast:y:0*
T0*'
_output_shapes
:          U
pattern/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :о
pattern/concatConcatV2salt_seq_2/dropout/Mul_1:z:0qty_seq_2/dropout/Mul_1:z:0pattern/concat/axis:output:0*
N*
T0*'
_output_shapes
:         @И
salt_pred/MatMul/ReadVariableOpReadVariableOp(salt_pred_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0О
salt_pred/MatMulMatMulpattern/concat:output:0'salt_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 salt_pred/BiasAdd/ReadVariableOpReadVariableOp)salt_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
salt_pred/BiasAddBiasAddsalt_pred/MatMul:product:0(salt_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         i
IdentityIdentitysalt_pred/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ├
NoOpNoOp,^qty_seq/lstm_cell_31/BiasAdd/ReadVariableOp+^qty_seq/lstm_cell_31/MatMul/ReadVariableOp-^qty_seq/lstm_cell_31/MatMul_1/ReadVariableOp^qty_seq/while!^salt_pred/BiasAdd/ReadVariableOp ^salt_pred/MatMul/ReadVariableOp-^salt_seq/lstm_cell_30/BiasAdd/ReadVariableOp,^salt_seq/lstm_cell_30/MatMul/ReadVariableOp.^salt_seq/lstm_cell_30/MatMul_1/ReadVariableOp^salt_seq/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         :         : : : : : : : : 2Z
+qty_seq/lstm_cell_31/BiasAdd/ReadVariableOp+qty_seq/lstm_cell_31/BiasAdd/ReadVariableOp2X
*qty_seq/lstm_cell_31/MatMul/ReadVariableOp*qty_seq/lstm_cell_31/MatMul/ReadVariableOp2\
,qty_seq/lstm_cell_31/MatMul_1/ReadVariableOp,qty_seq/lstm_cell_31/MatMul_1/ReadVariableOp2
qty_seq/whileqty_seq/while2D
 salt_pred/BiasAdd/ReadVariableOp salt_pred/BiasAdd/ReadVariableOp2B
salt_pred/MatMul/ReadVariableOpsalt_pred/MatMul/ReadVariableOp2\
,salt_seq/lstm_cell_30/BiasAdd/ReadVariableOp,salt_seq/lstm_cell_30/BiasAdd/ReadVariableOp2Z
+salt_seq/lstm_cell_30/MatMul/ReadVariableOp+salt_seq/lstm_cell_30/MatMul/ReadVariableOp2^
-salt_seq/lstm_cell_30/MatMul_1/ReadVariableOp-salt_seq/lstm_cell_30/MatMul_1/ReadVariableOp2 
salt_seq/whilesalt_seq/while:U Q
+
_output_shapes
:         
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/1
ы
Ў
-__inference_lstm_cell_30_layer_call_fn_129182

inputs
states_0
states_1
unknown:	А
	unknown_0:	 А
	unknown_1:	А
identity

identity_1

identity_2ИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_125761o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
йJ
Ь
D__inference_salt_seq_layer_call_and_return_conditional_losses_128303

inputs>
+lstm_cell_30_matmul_readvariableop_resource:	А@
-lstm_cell_30_matmul_1_readvariableop_resource:	 А;
,lstm_cell_30_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_30/BiasAdd/ReadVariableOpв"lstm_cell_30/MatMul/ReadVariableOpв$lstm_cell_30/MatMul_1/ReadVariableOpвwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskП
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АМ
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АН
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А^
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▌
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitn
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*'
_output_shapes
:          p
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*'
_output_shapes
:          w
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          h
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*'
_output_shapes
:          Ж
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          {
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          p
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*'
_output_shapes
:          e
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          К
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_128219*
condR
while_cond_128218*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          └
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ж
ш
A__inference_model_layer_call_and_return_conditional_losses_126597

inputs
inputs_1!
qty_seq_126400:	А!
qty_seq_126402:	 А
qty_seq_126404:	А"
salt_seq_126550:	А"
salt_seq_126552:	 А
salt_seq_126554:	А"
salt_pred_126591:@
salt_pred_126593:
identityИвqty_seq/StatefulPartitionedCallв!salt_pred/StatefulPartitionedCallв salt_seq/StatefulPartitionedCallА
qty_seq/StatefulPartitionedCallStatefulPartitionedCallinputs_1qty_seq_126400qty_seq_126402qty_seq_126404*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_126399Г
 salt_seq/StatefulPartitionedCallStatefulPartitionedCallinputssalt_seq_126550salt_seq_126552salt_seq_126554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_126549▀
salt_seq_2/PartitionedCallPartitionedCall)salt_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_126562▄
qty_seq_2/PartitionedCallPartitionedCall(qty_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_126569°
pattern/PartitionedCallPartitionedCall#salt_seq_2/PartitionedCall:output:0"qty_seq_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_126578О
!salt_pred/StatefulPartitionedCallStatefulPartitionedCall pattern/PartitionedCall:output:0salt_pred_126591salt_pred_126593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_126590y
IdentityIdentity*salt_pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         п
NoOpNoOp ^qty_seq/StatefulPartitionedCall"^salt_pred/StatefulPartitionedCall!^salt_seq/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         :         : : : : : : : : 2B
qty_seq/StatefulPartitionedCallqty_seq/StatefulPartitionedCall2F
!salt_pred/StatefulPartitionedCall!salt_pred/StatefulPartitionedCall2D
 salt_seq/StatefulPartitionedCall salt_seq/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:SO
+
_output_shapes
:         
 
_user_specified_nameinputs
щ

╒
&__inference_model_layer_call_fn_127102
	salt_data
quantity_data
unknown:	А
	unknown_0:	 А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	 А
	unknown_4:	А
	unknown_5:@
	unknown_6:
identityИвStatefulPartitionedCall╖
StatefulPartitionedCallStatefulPartitionedCall	salt_dataquantity_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_127061o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         :         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:         
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:         
'
_user_specified_namequantity_data
╒
Д
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_126111

inputs

states
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╢
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          N
ReluRelusplit:output:2*
T0*'
_output_shapes
:          _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates
А

e
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_129089

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ш
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:          Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ы
Ў
-__inference_lstm_cell_31_layer_call_fn_129280

inputs
states_0
states_1
unknown:	А
	unknown_0:	 А
	unknown_1:	А
identity

identity_1

identity_2ИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_126111o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
 	
d
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_126653

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ш
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:          Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
йJ
Ь
D__inference_salt_seq_layer_call_and_return_conditional_losses_126549

inputs>
+lstm_cell_30_matmul_readvariableop_resource:	А@
-lstm_cell_30_matmul_1_readvariableop_resource:	 А;
,lstm_cell_30_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_30/BiasAdd/ReadVariableOpв"lstm_cell_30/MatMul/ReadVariableOpв$lstm_cell_30/MatMul_1/ReadVariableOpвwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:          R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:          c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskП
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Р
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АМ
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         АН
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А^
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :▌
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitn
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*'
_output_shapes
:          p
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*'
_output_shapes
:          w
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          h
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*'
_output_shapes
:          Ж
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          {
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          p
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*'
_output_shapes
:          e
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          К
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : В
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_126465*
condR
while_cond_126464*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:          └
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
а
Ё
A__inference_model_layer_call_and_return_conditional_losses_127129
	salt_data
quantity_data!
qty_seq_127106:	А!
qty_seq_127108:	 А
qty_seq_127110:	А"
salt_seq_127113:	А"
salt_seq_127115:	 А
salt_seq_127117:	А"
salt_pred_127123:@
salt_pred_127125:
identityИвqty_seq/StatefulPartitionedCallв!salt_pred/StatefulPartitionedCallв salt_seq/StatefulPartitionedCallЕ
qty_seq/StatefulPartitionedCallStatefulPartitionedCallquantity_dataqty_seq_127106qty_seq_127108qty_seq_127110*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_qty_seq_layer_call_and_return_conditional_losses_126399Ж
 salt_seq/StatefulPartitionedCallStatefulPartitionedCall	salt_datasalt_seq_127113salt_seq_127115salt_seq_127117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_salt_seq_layer_call_and_return_conditional_losses_126549▀
salt_seq_2/PartitionedCallPartitionedCall)salt_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_126562▄
qty_seq_2/PartitionedCallPartitionedCall(qty_seq/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_126569°
pattern/PartitionedCallPartitionedCall#salt_seq_2/PartitionedCall:output:0"qty_seq_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_pattern_layer_call_and_return_conditional_losses_126578О
!salt_pred/StatefulPartitionedCallStatefulPartitionedCall pattern/PartitionedCall:output:0salt_pred_127123salt_pred_127125*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_salt_pred_layer_call_and_return_conditional_losses_126590y
IdentityIdentity*salt_pred/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         п
NoOpNoOp ^qty_seq/StatefulPartitionedCall"^salt_pred/StatefulPartitionedCall!^salt_seq/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         :         : : : : : : : : 2B
qty_seq/StatefulPartitionedCallqty_seq/StatefulPartitionedCall2F
!salt_pred/StatefulPartitionedCall!salt_pred/StatefulPartitionedCall2D
 salt_seq/StatefulPartitionedCall salt_seq/StatefulPartitionedCall:V R
+
_output_shapes
:         
#
_user_specified_name	salt_data:ZV
+
_output_shapes
:         
'
_user_specified_namequantity_data
Я8
╨
while_body_128692
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_31_matmul_readvariableop_resource_0:	АH
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:	 АC
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_31_matmul_readvariableop_resource:	АF
3while_lstm_cell_31_matmul_1_readvariableop_resource:	 АA
2while_lstm_cell_31_biasadd_readvariableop_resource:	АИв)while/lstm_cell_31/BiasAdd/ReadVariableOpв(while/lstm_cell_31/MatMul/ReadVariableOpв*while/lstm_cell_31/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Э
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0║
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0б
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АЫ
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0з
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitz
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*'
_output_shapes
:          |
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*'
_output_shapes
:          Ж
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          t
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ш
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          Н
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          |
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*'
_output_shapes
:          q
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          Ь
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥y
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          y
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          ╨

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
юB
Ё

salt_seq_while_body_127698.
*salt_seq_while_salt_seq_while_loop_counter4
0salt_seq_while_salt_seq_while_maximum_iterations
salt_seq_while_placeholder 
salt_seq_while_placeholder_1 
salt_seq_while_placeholder_2 
salt_seq_while_placeholder_3-
)salt_seq_while_salt_seq_strided_slice_1_0i
esalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0O
<salt_seq_while_lstm_cell_30_matmul_readvariableop_resource_0:	АQ
>salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource_0:	 АL
=salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource_0:	А
salt_seq_while_identity
salt_seq_while_identity_1
salt_seq_while_identity_2
salt_seq_while_identity_3
salt_seq_while_identity_4
salt_seq_while_identity_5+
'salt_seq_while_salt_seq_strided_slice_1g
csalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensorM
:salt_seq_while_lstm_cell_30_matmul_readvariableop_resource:	АO
<salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource:	 АJ
;salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource:	АИв2salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOpв1salt_seq/while/lstm_cell_30/MatMul/ReadVariableOpв3salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOpС
@salt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╙
2salt_seq/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemesalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0salt_seq_while_placeholderIsalt_seq/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0п
1salt_seq/while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp<salt_seq_while_lstm_cell_30_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0╒
"salt_seq/while/lstm_cell_30/MatMulMatMul9salt_seq/while/TensorArrayV2Read/TensorListGetItem:item:09salt_seq/while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А│
3salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp>salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0╝
$salt_seq/while/lstm_cell_30/MatMul_1MatMulsalt_seq_while_placeholder_2;salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╣
salt_seq/while/lstm_cell_30/addAddV2,salt_seq/while/lstm_cell_30/MatMul:product:0.salt_seq/while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:         Ан
2salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp=salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0┬
#salt_seq/while/lstm_cell_30/BiasAddBiasAdd#salt_seq/while/lstm_cell_30/add:z:0:salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
+salt_seq/while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :К
!salt_seq/while/lstm_cell_30/splitSplit4salt_seq/while/lstm_cell_30/split/split_dim:output:0,salt_seq/while/lstm_cell_30/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitМ
#salt_seq/while/lstm_cell_30/SigmoidSigmoid*salt_seq/while/lstm_cell_30/split:output:0*
T0*'
_output_shapes
:          О
%salt_seq/while/lstm_cell_30/Sigmoid_1Sigmoid*salt_seq/while/lstm_cell_30/split:output:1*
T0*'
_output_shapes
:          б
salt_seq/while/lstm_cell_30/mulMul)salt_seq/while/lstm_cell_30/Sigmoid_1:y:0salt_seq_while_placeholder_3*
T0*'
_output_shapes
:          Ж
 salt_seq/while/lstm_cell_30/ReluRelu*salt_seq/while/lstm_cell_30/split:output:2*
T0*'
_output_shapes
:          │
!salt_seq/while/lstm_cell_30/mul_1Mul'salt_seq/while/lstm_cell_30/Sigmoid:y:0.salt_seq/while/lstm_cell_30/Relu:activations:0*
T0*'
_output_shapes
:          и
!salt_seq/while/lstm_cell_30/add_1AddV2#salt_seq/while/lstm_cell_30/mul:z:0%salt_seq/while/lstm_cell_30/mul_1:z:0*
T0*'
_output_shapes
:          О
%salt_seq/while/lstm_cell_30/Sigmoid_2Sigmoid*salt_seq/while/lstm_cell_30/split:output:3*
T0*'
_output_shapes
:          Г
"salt_seq/while/lstm_cell_30/Relu_1Relu%salt_seq/while/lstm_cell_30/add_1:z:0*
T0*'
_output_shapes
:          ╖
!salt_seq/while/lstm_cell_30/mul_2Mul)salt_seq/while/lstm_cell_30/Sigmoid_2:y:00salt_seq/while/lstm_cell_30/Relu_1:activations:0*
T0*'
_output_shapes
:          щ
3salt_seq/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemsalt_seq_while_placeholder_1salt_seq_while_placeholder%salt_seq/while/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥V
salt_seq/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
salt_seq/while/addAddV2salt_seq_while_placeholdersalt_seq/while/add/y:output:0*
T0*
_output_shapes
: X
salt_seq/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
salt_seq/while/add_1AddV2*salt_seq_while_salt_seq_while_loop_countersalt_seq/while/add_1/y:output:0*
T0*
_output_shapes
: t
salt_seq/while/IdentityIdentitysalt_seq/while/add_1:z:0^salt_seq/while/NoOp*
T0*
_output_shapes
: О
salt_seq/while/Identity_1Identity0salt_seq_while_salt_seq_while_maximum_iterations^salt_seq/while/NoOp*
T0*
_output_shapes
: t
salt_seq/while/Identity_2Identitysalt_seq/while/add:z:0^salt_seq/while/NoOp*
T0*
_output_shapes
: ┤
salt_seq/while/Identity_3IdentityCsalt_seq/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^salt_seq/while/NoOp*
T0*
_output_shapes
: :щш╥Ф
salt_seq/while/Identity_4Identity%salt_seq/while/lstm_cell_30/mul_2:z:0^salt_seq/while/NoOp*
T0*'
_output_shapes
:          Ф
salt_seq/while/Identity_5Identity%salt_seq/while/lstm_cell_30/add_1:z:0^salt_seq/while/NoOp*
T0*'
_output_shapes
:          Ї
salt_seq/while/NoOpNoOp3^salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOp2^salt_seq/while/lstm_cell_30/MatMul/ReadVariableOp4^salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
salt_seq_while_identity salt_seq/while/Identity:output:0"?
salt_seq_while_identity_1"salt_seq/while/Identity_1:output:0"?
salt_seq_while_identity_2"salt_seq/while/Identity_2:output:0"?
salt_seq_while_identity_3"salt_seq/while/Identity_3:output:0"?
salt_seq_while_identity_4"salt_seq/while/Identity_4:output:0"?
salt_seq_while_identity_5"salt_seq/while/Identity_5:output:0"|
;salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource=salt_seq_while_lstm_cell_30_biasadd_readvariableop_resource_0"~
<salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource>salt_seq_while_lstm_cell_30_matmul_1_readvariableop_resource_0"z
:salt_seq_while_lstm_cell_30_matmul_readvariableop_resource<salt_seq_while_lstm_cell_30_matmul_readvariableop_resource_0"T
'salt_seq_while_salt_seq_strided_slice_1)salt_seq_while_salt_seq_strided_slice_1_0"╠
csalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensoresalt_seq_while_tensorarrayv2read_tensorlistgetitem_salt_seq_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2h
2salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOp2salt_seq/while/lstm_cell_30/BiasAdd/ReadVariableOp2f
1salt_seq/while/lstm_cell_30/MatMul/ReadVariableOp1salt_seq/while/lstm_cell_30/MatMul/ReadVariableOp2j
3salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOp3salt_seq/while/lstm_cell_30/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
 	
d
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_129116

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ш
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:          Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
∙"
у
while_body_126170
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_31_126194_0:	А.
while_lstm_cell_31_126196_0:	 А*
while_lstm_cell_31_126198_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_31_126194:	А,
while_lstm_cell_31_126196:	 А(
while_lstm_cell_31_126198:	АИв*while/lstm_cell_31/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0│
*while/lstm_cell_31/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_31_126194_0while_lstm_cell_31_126196_0while_lstm_cell_31_126198_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_126111▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_31/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥Р
while/Identity_4Identity3while/lstm_cell_31/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          Р
while/Identity_5Identity3while/lstm_cell_31/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          y

while/NoOpNoOp+^while/lstm_cell_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_31_126194while_lstm_cell_31_126194_0"8
while_lstm_cell_31_126196while_lstm_cell_31_126196_0"8
while_lstm_cell_31_126198while_lstm_cell_31_126198_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2X
*while/lstm_cell_31/StatefulPartitionedCall*while/lstm_cell_31/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
Я8
╨
while_body_128549
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_31_matmul_readvariableop_resource_0:	АH
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:	 АC
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_31_matmul_readvariableop_resource:	АF
3while_lstm_cell_31_matmul_1_readvariableop_resource:	 АA
2while_lstm_cell_31_biasadd_readvariableop_resource:	АИв)while/lstm_cell_31/BiasAdd/ReadVariableOpв(while/lstm_cell_31/MatMul/ReadVariableOpв*while/lstm_cell_31/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Э
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype0║
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype0б
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЮ
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:         АЫ
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype0з
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :я
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_splitz
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*'
_output_shapes
:          |
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*'
_output_shapes
:          Ж
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          t
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*'
_output_shapes
:          Ш
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*'
_output_shapes
:          Н
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*'
_output_shapes
:          |
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*'
_output_shapes
:          q
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*'
_output_shapes
:          Ь
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*'
_output_shapes
:          ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Щ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :щш╥y
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          y
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          ╨

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: "█L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Б
serving_defaultэ
K
quantity_data:
serving_default_quantity_data:0         
C
	salt_data6
serving_default_salt_data:0         =
	salt_pred0
StatefulPartitionedCall:0         tensorflow/serving/predict:ю╚
Щ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
┌
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
┌
cell

state_spec
	variables
trainable_variables
regularization_losses
 	keras_api
!_random_generator
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
╝
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(_random_generator
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
е
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
є
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate8mР9mСEmТFmУGmФHmХImЦJmЧ8vШ9vЩEvЪFvЫGvЬHvЭIvЮJvЯ"
	optimizer
X
E0
F1
G2
H3
I4
J5
86
97"
trackable_list_wrapper
X
E0
F1
G2
H3
I4
J5
86
97"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ц2у
&__inference_model_layer_call_fn_126616
&__inference_model_layer_call_fn_127184
&__inference_model_layer_call_fn_127206
&__inference_model_layer_call_fn_127102└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
A__inference_model_layer_call_and_return_conditional_losses_127499
A__inference_model_layer_call_and_return_conditional_losses_127806
A__inference_model_layer_call_and_return_conditional_losses_127129
A__inference_model_layer_call_and_return_conditional_losses_127156└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▌B┌
!__inference__wrapped_model_125548	salt_dataquantity_data"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
,
Pserving_default"
signature_map
°
Q
state_size

Ekernel
Frecurrent_kernel
Gbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V_random_generator
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

Ystates
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
З2Д
)__inference_salt_seq_layer_call_fn_127841
)__inference_salt_seq_layer_call_fn_127852
)__inference_salt_seq_layer_call_fn_127863
)__inference_salt_seq_layer_call_fn_127874╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
є2Ё
D__inference_salt_seq_layer_call_and_return_conditional_losses_128017
D__inference_salt_seq_layer_call_and_return_conditional_losses_128160
D__inference_salt_seq_layer_call_and_return_conditional_losses_128303
D__inference_salt_seq_layer_call_and_return_conditional_losses_128446╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
°
_
state_size

Hkernel
Irecurrent_kernel
Jbias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d_random_generator
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

gstates
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Г2А
(__inference_qty_seq_layer_call_fn_128457
(__inference_qty_seq_layer_call_fn_128468
(__inference_qty_seq_layer_call_fn_128479
(__inference_qty_seq_layer_call_fn_128490╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
я2ь
C__inference_qty_seq_layer_call_and_return_conditional_losses_128633
C__inference_qty_seq_layer_call_and_return_conditional_losses_128776
C__inference_qty_seq_layer_call_and_return_conditional_losses_128919
C__inference_qty_seq_layer_call_and_return_conditional_losses_129062╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
$	variables
%trainable_variables
&regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ф2С
+__inference_salt_seq_2_layer_call_fn_129067
+__inference_salt_seq_2_layer_call_fn_129072┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_129077
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_129089┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Т2П
*__inference_qty_seq_2_layer_call_fn_129094
*__inference_qty_seq_2_layer_call_fn_129099┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╚2┼
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_129104
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_129116┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
╥2╧
(__inference_pattern_layer_call_fn_129122в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_pattern_layer_call_and_return_conditional_losses_129129в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
": @2salt_pred/kernel
:2salt_pred/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
о
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
Аlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
╘2╤
*__inference_salt_pred_layer_call_fn_129138в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_salt_pred_layer_call_and_return_conditional_losses_129148в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
/:-	А2salt_seq/lstm_cell_30/kernel
9:7	 А2&salt_seq/lstm_cell_30/recurrent_kernel
):'А2salt_seq/lstm_cell_30/bias
.:,	А2qty_seq/lstm_cell_31/kernel
8:6	 А2%qty_seq/lstm_cell_31/recurrent_kernel
(:&А2qty_seq/lstm_cell_31/bias
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
(
Б0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
$__inference_signature_wrapper_127830quantity_data	salt_data"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
в2Я
-__inference_lstm_cell_30_layer_call_fn_129165
-__inference_lstm_cell_30_layer_call_fn_129182╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╪2╒
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_129214
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_129246╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
`	variables
atrainable_variables
bregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
в2Я
-__inference_lstm_cell_31_layer_call_fn_129263
-__inference_lstm_cell_31_layer_call_fn_129280╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╪2╒
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_129312
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_129344╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

Мtotal

Нcount
О	variables
П	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
М0
Н1"
trackable_list_wrapper
.
О	variables"
_generic_user_object
':%@2Adam/salt_pred/kernel/m
!:2Adam/salt_pred/bias/m
4:2	А2#Adam/salt_seq/lstm_cell_30/kernel/m
>:<	 А2-Adam/salt_seq/lstm_cell_30/recurrent_kernel/m
.:,А2!Adam/salt_seq/lstm_cell_30/bias/m
3:1	А2"Adam/qty_seq/lstm_cell_31/kernel/m
=:;	 А2,Adam/qty_seq/lstm_cell_31/recurrent_kernel/m
-:+А2 Adam/qty_seq/lstm_cell_31/bias/m
':%@2Adam/salt_pred/kernel/v
!:2Adam/salt_pred/bias/v
4:2	А2#Adam/salt_seq/lstm_cell_30/kernel/v
>:<	 А2-Adam/salt_seq/lstm_cell_30/recurrent_kernel/v
.:,А2!Adam/salt_seq/lstm_cell_30/bias/v
3:1	А2"Adam/qty_seq/lstm_cell_31/kernel/v
=:;	 А2,Adam/qty_seq/lstm_cell_31/recurrent_kernel/v
-:+А2 Adam/qty_seq/lstm_cell_31/bias/v╤
!__inference__wrapped_model_125548лHIJEFG89hвe
^в[
YвV
'К$
	salt_data         
+К(
quantity_data         
к "5к2
0
	salt_pred#К 
	salt_pred         ╩
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_129214¤EFGАв}
vвs
 К
inputs         
KвH
"К
states/0          
"К
states/1          
p 
к "sвp
iвf
К
0/0          
EЪB
К
0/1/0          
К
0/1/1          
Ъ ╩
H__inference_lstm_cell_30_layer_call_and_return_conditional_losses_129246¤EFGАв}
vвs
 К
inputs         
KвH
"К
states/0          
"К
states/1          
p
к "sвp
iвf
К
0/0          
EЪB
К
0/1/0          
К
0/1/1          
Ъ Я
-__inference_lstm_cell_30_layer_call_fn_129165эEFGАв}
vвs
 К
inputs         
KвH
"К
states/0          
"К
states/1          
p 
к "cв`
К
0          
AЪ>
К
1/0          
К
1/1          Я
-__inference_lstm_cell_30_layer_call_fn_129182эEFGАв}
vвs
 К
inputs         
KвH
"К
states/0          
"К
states/1          
p
к "cв`
К
0          
AЪ>
К
1/0          
К
1/1          ╩
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_129312¤HIJАв}
vвs
 К
inputs         
KвH
"К
states/0          
"К
states/1          
p 
к "sвp
iвf
К
0/0          
EЪB
К
0/1/0          
К
0/1/1          
Ъ ╩
H__inference_lstm_cell_31_layer_call_and_return_conditional_losses_129344¤HIJАв}
vвs
 К
inputs         
KвH
"К
states/0          
"К
states/1          
p
к "sвp
iвf
К
0/0          
EЪB
К
0/1/0          
К
0/1/1          
Ъ Я
-__inference_lstm_cell_31_layer_call_fn_129263эHIJАв}
vвs
 К
inputs         
KвH
"К
states/0          
"К
states/1          
p 
к "cв`
К
0          
AЪ>
К
1/0          
К
1/1          Я
-__inference_lstm_cell_31_layer_call_fn_129280эHIJАв}
vвs
 К
inputs         
KвH
"К
states/0          
"К
states/1          
p
к "cв`
К
0          
AЪ>
К
1/0          
К
1/1          щ
A__inference_model_layer_call_and_return_conditional_losses_127129гHIJEFG89pвm
fвc
YвV
'К$
	salt_data         
+К(
quantity_data         
p 

 
к "%в"
К
0         
Ъ щ
A__inference_model_layer_call_and_return_conditional_losses_127156гHIJEFG89pвm
fвc
YвV
'К$
	salt_data         
+К(
quantity_data         
p

 
к "%в"
К
0         
Ъ у
A__inference_model_layer_call_and_return_conditional_losses_127499ЭHIJEFG89jвg
`в]
SвP
&К#
inputs/0         
&К#
inputs/1         
p 

 
к "%в"
К
0         
Ъ у
A__inference_model_layer_call_and_return_conditional_losses_127806ЭHIJEFG89jвg
`в]
SвP
&К#
inputs/0         
&К#
inputs/1         
p

 
к "%в"
К
0         
Ъ ┴
&__inference_model_layer_call_fn_126616ЦHIJEFG89pвm
fвc
YвV
'К$
	salt_data         
+К(
quantity_data         
p 

 
к "К         ┴
&__inference_model_layer_call_fn_127102ЦHIJEFG89pвm
fвc
YвV
'К$
	salt_data         
+К(
quantity_data         
p

 
к "К         ╗
&__inference_model_layer_call_fn_127184РHIJEFG89jвg
`в]
SвP
&К#
inputs/0         
&К#
inputs/1         
p 

 
к "К         ╗
&__inference_model_layer_call_fn_127206РHIJEFG89jвg
`в]
SвP
&К#
inputs/0         
&К#
inputs/1         
p

 
к "К         ╦
C__inference_pattern_layer_call_and_return_conditional_losses_129129ГZвW
PвM
KЪH
"К
inputs/0          
"К
inputs/1          
к "%в"
К
0         @
Ъ в
(__inference_pattern_layer_call_fn_129122vZвW
PвM
KЪH
"К
inputs/0          
"К
inputs/1          
к "К         @е
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_129104\3в0
)в&
 К
inputs          
p 
к "%в"
К
0          
Ъ е
E__inference_qty_seq_2_layer_call_and_return_conditional_losses_129116\3в0
)в&
 К
inputs          
p
к "%в"
К
0          
Ъ }
*__inference_qty_seq_2_layer_call_fn_129094O3в0
)в&
 К
inputs          
p 
к "К          }
*__inference_qty_seq_2_layer_call_fn_129099O3в0
)в&
 К
inputs          
p
к "К          ─
C__inference_qty_seq_layer_call_and_return_conditional_losses_128633}HIJOвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "%в"
К
0          
Ъ ─
C__inference_qty_seq_layer_call_and_return_conditional_losses_128776}HIJOвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "%в"
К
0          
Ъ ┤
C__inference_qty_seq_layer_call_and_return_conditional_losses_128919mHIJ?в<
5в2
$К!
inputs         

 
p 

 
к "%в"
К
0          
Ъ ┤
C__inference_qty_seq_layer_call_and_return_conditional_losses_129062mHIJ?в<
5в2
$К!
inputs         

 
p

 
к "%в"
К
0          
Ъ Ь
(__inference_qty_seq_layer_call_fn_128457pHIJOвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "К          Ь
(__inference_qty_seq_layer_call_fn_128468pHIJOвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "К          М
(__inference_qty_seq_layer_call_fn_128479`HIJ?в<
5в2
$К!
inputs         

 
p 

 
к "К          М
(__inference_qty_seq_layer_call_fn_128490`HIJ?в<
5в2
$К!
inputs         

 
p

 
к "К          е
E__inference_salt_pred_layer_call_and_return_conditional_losses_129148\89/в,
%в"
 К
inputs         @
к "%в"
К
0         
Ъ }
*__inference_salt_pred_layer_call_fn_129138O89/в,
%в"
 К
inputs         @
к "К         ж
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_129077\3в0
)в&
 К
inputs          
p 
к "%в"
К
0          
Ъ ж
F__inference_salt_seq_2_layer_call_and_return_conditional_losses_129089\3в0
)в&
 К
inputs          
p
к "%в"
К
0          
Ъ ~
+__inference_salt_seq_2_layer_call_fn_129067O3в0
)в&
 К
inputs          
p 
к "К          ~
+__inference_salt_seq_2_layer_call_fn_129072O3в0
)в&
 К
inputs          
p
к "К          ┼
D__inference_salt_seq_layer_call_and_return_conditional_losses_128017}EFGOвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "%в"
К
0          
Ъ ┼
D__inference_salt_seq_layer_call_and_return_conditional_losses_128160}EFGOвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "%в"
К
0          
Ъ ╡
D__inference_salt_seq_layer_call_and_return_conditional_losses_128303mEFG?в<
5в2
$К!
inputs         

 
p 

 
к "%в"
К
0          
Ъ ╡
D__inference_salt_seq_layer_call_and_return_conditional_losses_128446mEFG?в<
5в2
$К!
inputs         

 
p

 
к "%в"
К
0          
Ъ Э
)__inference_salt_seq_layer_call_fn_127841pEFGOвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "К          Э
)__inference_salt_seq_layer_call_fn_127852pEFGOвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "К          Н
)__inference_salt_seq_layer_call_fn_127863`EFG?в<
5в2
$К!
inputs         

 
p 

 
к "К          Н
)__inference_salt_seq_layer_call_fn_127874`EFG?в<
5в2
$К!
inputs         

 
p

 
к "К          ю
$__inference_signature_wrapper_127830┼HIJEFG89Бв~
в 
wкt
<
quantity_data+К(
quantity_data         
4
	salt_data'К$
	salt_data         "5к2
0
	salt_pred#К 
	salt_pred         