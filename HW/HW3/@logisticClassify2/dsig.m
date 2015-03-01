function value = dsig( z )
%derivative of sigmod function
value = sig(z) .* (1-sig(z));

end

